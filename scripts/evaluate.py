#!/usr/bin/env python3

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def load_model(model_path, device):
    """Load the fine-tuned model"""
    # Check if model path is a checkpoint or a HuggingFace model ID
    if os.path.isfile(model_path):
        print(f"Loading fine-tuned model from checkpoint: {model_path}")
        # First load base model architecture
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Load fine-tuned weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Loading model directly: {model_path}")
        model = GPT2LMHeadModel.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model

def format_patient_data(row):
    """Convert a patient data row to a text format for evaluation"""
    # Start building patient description - same format as used in training
    text = "Patient: "
    
    # Add demographic and health metrics (dynamically based on columns available)
    metrics = []
    for col in row.index:
        if col.lower() != 'highriskflag' and col.lower() != 'patientid' and not pd.isna(row[col]):
            # Format the column name to be more readable
            readable_col = ' '.join(word.capitalize() for word in re.split(r'(?=[A-Z])', col) if word)
            
            # Format the value appropriately
            if isinstance(row[col], bool) or (isinstance(row[col], (int, float)) and (row[col] == 0 or row[col] == 1)):
                # Boolean values
                value = "Yes" if row[col] else "No"
            else:
                # Regular values
                value = str(row[col])
            
            metrics.append(f"{readable_col} {value}")
    
    text += ", ".join(metrics) + "."
    
    # For prediction, we add the prompt but not the answer
    text += " Risk:"
    
    return text

def extract_risk_label(text):
    """Extract risk label from generated text"""
    # Look for the pattern "Risk: X" where X could be "HighRisk" or "LowRisk"
    match = re.search(r'Risk:\s*(\w+)', text)
    if match:
        label = match.group(1).strip()
        if 'high' in label.lower():
            return 'HighRisk'
        elif 'low' in label.lower():
            return 'LowRisk'
    
    # If no clear match, check for any related words in the response
    if 'high' in text.lower() and 'risk' in text.lower():
        return 'HighRisk'
    elif 'low' in text.lower() and 'risk' in text.lower():
        return 'LowRisk'
    
    # Default fallback
    return 'Unknown'

def get_true_risk_label(row):
    """Get the true risk label from a data row"""
    if 'highriskflag' in [col.lower() for col in row.index]:
        risk_col = next(col for col in row.index if col.lower() == 'highriskflag')
        risk_value = row[risk_col]
        if isinstance(risk_value, (int, float)):
            return "HighRisk" if risk_value == 1 else "LowRisk"
        else:
            return "HighRisk" if str(risk_value).lower() in ['true', 'yes', 'high', '1'] else "LowRisk"
    return "Unknown"

def evaluate_csv(model, tokenizer, csv_path, device, max_length=512):
    """Evaluate the model on a CSV file with medical data"""
    print(f"Evaluating model on {csv_path}")
    
    # Load test data
    df = pd.read_csv(csv_path)
    
    # Initialize lists for predictions and ground truth
    predictions = []
    true_labels = []
    prediction_texts = []
    input_texts = []
    
    # Batch processing for efficiency
    batch_size = 8
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        batch_inputs = []
        batch_true_labels = []
        
        for _, row in batch_df.iterrows():
            # Format input text
            input_text = format_patient_data(row)
            batch_inputs.append(input_text)
            input_texts.append(input_text)
            
            # Get true label
            true_label = get_true_risk_label(row)
            batch_true_labels.append(true_label)
            true_labels.append(true_label)
        
        # Tokenize all inputs in the batch
        encoded_inputs = tokenizer(batch_inputs, return_tensors="pt", max_length=max_length, 
                                  truncation=True, padding=True)
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                max_length=max_length + 20,  # Allow for response generation
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Use greedy decoding for deterministic results
            )
        
        # Process each generated output
        for output_ids in outputs:
            # Decode generated text
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            prediction_texts.append(generated_text)
            
            # Extract risk prediction from generated text
            predicted_label = extract_risk_label(generated_text)
            predictions.append(predicted_label)
        
        # Print progress
        print(f"Processed {min(i+batch_size, len(df))}/{len(df)} samples", end="\r")
    
    print("\nProcessing complete, calculating metrics...")
    
    # Filter out any "Unknown" predictions for fair metric calculation
    valid_indices = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) 
                     if p != "Unknown" and t != "Unknown"]
    
    if not valid_indices:
        print("Warning: No valid predictions found, unable to calculate metrics.")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "samples_evaluated": len(df),
            "error_rate": 1.0
        }
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    
    # Convert string labels to binary for metric calculation
    binary_predictions = [1 if p == "HighRisk" else 0 for p in valid_predictions]
    binary_true_labels = [1 if t == "HighRisk" else 0 for t in valid_true_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(binary_true_labels, binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_true_labels, binary_predictions, average='binary', pos_label=1
    )
    
    # Create confusion matrix
    cm = confusion_matrix(binary_true_labels, binary_predictions)
    
    # Calculate class-specific metrics
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),  # For HighRisk class
        "recall": float(recall),        # For HighRisk class
        "f1": float(f1),                # For HighRisk class
        "samples_evaluated": len(df),
        "valid_samples": len(valid_indices),
        "unknown_predictions": len(df) - len(valid_indices),
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
        "error_rate": 1.0 - float(accuracy)
    }
    
    # Save sample predictions for review
    sample_indices = valid_indices[:min(5, len(valid_indices))]
    samples = [
        {
            "input": input_texts[i],
            "true_label": true_labels[i],
            "prediction": predictions[i],
            "full_output": prediction_texts[i]
        }
        for i in sample_indices
    ]
    results["samples"] = samples
    
    return results

def plot_confusion_matrix(cm, model_path, save_dir=None):
    """Plot confusion matrix and save to file"""
    # Ensure output directory exists
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = "docs/summaries"
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["LowRisk", "HighRisk"],
               yticklabels=["LowRisk", "HighRisk"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure
    model_name = Path(model_path).stem
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = f"{save_dir}/{date_str}_{model_name}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def process_text_file(model, tokenizer, text_path, device, max_length=512):
    """Evaluate model's performance on research text"""
    print(f"Processing text file: {text_path}")
    
    # Load text file
    with open(text_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Chunk the text into sections for processing
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    
    for line in lines:
        current_chunk.append(line)
        if len(' '.join(current_chunk)) > 300:  # Process chunks of ~300 characters
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:  # Add the last chunk
        chunks.append(' '.join(current_chunk))
    
    # Process chunks
    summaries = []
    for i, chunk in enumerate(chunks[:5]):  # Process up to 5 chunks
        print(f"Processing chunk {i+1}/{min(len(chunks), 5)}")
        
        # Create a prompt for summarization
        prompt = f"Summarize the following medical information:\n{chunk}\n\nSummary:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length,
                         truncation=True, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length + 100,  # Allow room for summary
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,  # Use sampling for more natural text
                temperature=0.7,  # Control randomness
                top_p=0.9  # Nucleus sampling
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract summary (text after "Summary:")
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[1].strip()
        else:
            summary = generated_text.strip()
        
        summaries.append(summary)
    
    # Calculate perplexity on test text as a basic measure
    try:
        perplexity_samples = []
        # Sample a few sections for perplexity calculation
        for i in range(min(3, len(chunks))):
            sample = chunks[i]
            inputs = tokenizer(sample, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexity_samples.append(perplexity)
        
        avg_perplexity = sum(perplexity_samples) / len(perplexity_samples)
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        avg_perplexity = None
    
    return {
        "summaries": summaries,
        "perplexity": avg_perplexity,
        "chunks_processed": min(5, len(chunks))
    }

def generate_eval_summary(results, model_path, text_results=None):
    """Generate markdown summary of evaluation"""
    date_str = datetime.now().strftime("%Y%m%d")
    summary_path = Path("docs/summaries") / f"{date_str}_eval_summary.md"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, "w") as f:
        f.write(f"# Evaluation Summary {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- Model path: {model_path}\n")
        f.write(f"- Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Samples evaluated: {results['samples_evaluated']}\n")
        if 'valid_samples' in results:
            f.write(f"- Valid samples: {results['valid_samples']} ({results['valid_samples']/results['samples_evaluated']*100:.1f}%)\n")
        if 'unknown_predictions' in results and results['unknown_predictions'] > 0:
            f.write(f"- Unknown predictions: {results['unknown_predictions']} ({results['unknown_predictions']/results['samples_evaluated']*100:.1f}%)\n")
        f.write("\n")
        
        f.write("## Classification Metrics\n\n")
        
        f.write("| Metric    | Overall | HighRisk Class |\n")
        f.write("|-----------|---------|---------------|\n")
        f.write(f"| Accuracy  | {results['accuracy']:.4f} | - |\n")
        f.write(f"| Precision | - | {results['precision']:.4f} |\n")
        f.write(f"| Recall    | - | {results['recall']:.4f} |\n")
        f.write(f"| F1 Score  | - | {results['f1']:.4f} |\n")
        f.write("\n")
        
        # Add confusion matrix if available
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            if len(cm) == 2 and len(cm[0]) == 2:
                f.write("## Confusion Matrix\n\n")
                f.write("| Predicted → <br> Actual ↓ | LowRisk | HighRisk |\n")
                f.write("|---------------------------|---------|----------|\n")
                f.write(f"| LowRisk                   | {cm[0][0]}       | {cm[0][1]}        |\n")
                f.write(f"| HighRisk                  | {cm[1][0]}       | {cm[1][1]}        |\n")
                f.write("\n")
                
                # Generate and reference the confusion matrix plot
                try:
                    cm_plot_path = plot_confusion_matrix(np.array(cm), model_path)
                    rel_path = os.path.relpath(cm_plot_path, os.path.dirname(summary_path))
                    f.write(f"![Confusion Matrix]({rel_path})\n\n")
                except Exception as e:
                    f.write(f"*Error generating confusion matrix plot: {e}*\n\n")
        
        # Add sample predictions
        if 'samples' in results and results['samples']:
            f.write("## Sample Predictions\n\n")
            for i, sample in enumerate(results['samples']):
                f.write(f"### Sample {i+1}\n\n")
                f.write(f"**Input**: `{sample['input']}`\n\n")
                f.write(f"**True Label**: {sample['true_label']}\n\n")
                f.write(f"**Predicted**: {sample['prediction']}\n\n")
                f.write(f"**Full Model Output**: \n```\n{sample['full_output']}\n```\n\n")
        
        # Add text evaluation results if available
        if text_results:
            f.write("## Text Generation Quality\n\n")
            
            if 'perplexity' in text_results and text_results['perplexity'] is not None:
                f.write(f"- **Perplexity**: {text_results['perplexity']:.2f}\n")
            else:
                f.write("- **Perplexity**: Not calculated\n")
            
            f.write(f"- **Chunks processed**: {text_results['chunks_processed']}\n\n")
            
            if 'summaries' in text_results and text_results['summaries']:
                f.write("### Generated Summaries\n\n")
                for i, summary in enumerate(text_results['summaries']):
                    f.write(f"#### Summary {i+1}\n\n")
                    f.write(f"```\n{summary}\n```\n\n")
        
        # Add recommendations based on results
        f.write("## Recommendations\n\n")
        
        if results['accuracy'] < 0.7:
            f.write("- **Model Accuracy Needs Improvement**: Consider retraining with more data or adjusting hyperparameters.\n")
        
        # Class imbalance detection
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            if len(cm) == 2 and len(cm[0]) == 2:
                true_positives = cm[1][1]
                false_negatives = cm[1][0]
                if true_positives < false_negatives:
                    f.write("- **High-Risk Recall Issue**: The model is missing too many high-risk cases. Consider adjusting class weights during training.\n")
        
        # High error rate
        if results['error_rate'] > 0.3:
            f.write("- **High Error Rate**: Consider implementing a confidence threshold to filter out uncertain predictions.\n")
        
        f.write("\n## Generated by\n")
        f.write("Agent: Evaluator\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    print(f"Evaluation summary written to {summary_path}")
    return summary_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned GPT-2 model")
    
    parser.add_argument("--model_path", default="models/finetuned/best_model.pt", 
                       help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--test_csv", default="processed/test_data.csv",
                       help="Path to test data file (CSV)")
    parser.add_argument("--test_text", default=None,
                       help="Optional text file for evaluating generation quality")
    parser.add_argument("--output_file", default=None,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for processing")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path) and "gpt2" not in args.model_path:
        print(f"Error: Model path not found: {args.model_path}")
        print("Using default gpt2 model instead")
        args.model_path = "gpt2"
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Initialize results
    all_results = {}
    
    # Check that input files exist
    csv_results = None
    if os.path.exists(args.test_csv):
        # Evaluate classification
        csv_results = evaluate_csv(model, tokenizer, args.test_csv, device, args.max_length)
        all_results["classification"] = csv_results
        
        # Print results
        print("\nClassification Results:")
        print(f"Accuracy: {csv_results['accuracy']:.4f}")
        print(f"Precision (HighRisk): {csv_results['precision']:.4f}")
        print(f"Recall (HighRisk): {csv_results['recall']:.4f}")
        print(f"F1 Score (HighRisk): {csv_results['f1']:.4f}")
    else:
        print(f"Warning: Test CSV file not found: {args.test_csv}")
    
    # Evaluate text generation if provided
    text_results = None
    if args.test_text and os.path.exists(args.test_text):
        print("\nEvaluating text generation...")
        text_results = process_text_file(model, tokenizer, args.test_text, device, args.max_length)
        all_results["text_generation"] = text_results
        
        if 'perplexity' in text_results and text_results['perplexity'] is not None:
            print(f"Perplexity: {text_results['perplexity']:.2f}")
        
        print(f"Generated {len(text_results['summaries'])} summaries")
    elif args.test_text:
        print(f"Warning: Test text file not found: {args.test_text}")
    
    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nEvaluation results saved to {args.output_file}")
    
    # Generate evaluation summary
    if csv_results:
        summary_path = generate_eval_summary(csv_results, args.model_path, text_results)
        print(f"Evaluation summary written to {summary_path}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()