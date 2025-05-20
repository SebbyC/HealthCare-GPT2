#!/usr/bin/env python3

import os
import json
import argparse
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model(model_path):
    """Load the fine-tuned model"""
    global model, tokenizer, device
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model architecture
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Load fine-tuned weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health check"""
    if model is not None and tokenizer is not None:
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "model not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for text generation"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    input_text = data['text']
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 0.9)
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        "generated_text": generated_text,
        "input_text": input_text
    })

def main():
    parser = argparse.ArgumentParser(description="Serve fine-tuned GPT-2 model")
    
    parser.add_argument("--model_path", default="models/finetuned/best_model.pt", 
                       help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path)
    
    # Run server
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()