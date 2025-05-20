#!/usr/bin/env python3

import os
import sys
import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

class HealthcareDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load all text files in the given directory
        path = Path(data_path)
        for file_path in path.glob("*.txt"):
            with open(file_path, 'r') as f:
                text = f.read()
                self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:].clone()
        labels[-1] = -100  # Ignore last token prediction
        
        # Mask out padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train(args):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and args.fp16
    
    print(f"Using device: {device}, AMP: {use_amp}")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(args.base_model)
    model.to(device)
    
    # Create datasets
    train_dataset = HealthcareDataset(Path(args.data_dir) / "train", tokenizer, args.max_length)
    val_dataset = HealthcareDataset(Path(args.data_dir) / "val", tokenizer, args.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Create learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # Setup scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
            
            if step % args.logging_steps == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"  Validation loss: {avg_val_loss:.4f}")
        
        # Save statistics
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        training_stats.append(epoch_stats)
        
        # Save model checkpoint
        checkpoint_path = output_dir / f"epoch{epoch+1}_valLoss{avg_val_loss:.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
        # Update best model if needed
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_checkpoint_path)
        
        # Save latest model
        latest_checkpoint_path = output_dir / "latest.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, latest_checkpoint_path)
    
    # Save training stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f, indent=2)
    
    # Generate training summary
    generate_training_summary(training_stats, args)
    
    print("Training complete!")

def generate_training_summary(stats, args):
    """Generate markdown summary of training"""
    date_str = datetime.now().strftime("%Y%m%d")
    summary_path = Path("docs/summaries") / f"{date_str}_train_summary.md"
    
    with open(summary_path, "w") as f:
        f.write(f"# Training Summary {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Base model: {args.base_model}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.learning_rate}\n")
        f.write(f"- Epochs: {args.num_epochs}\n")
        f.write(f"- Max sequence length: {args.max_length}\n")
        f.write(f"- Mixed precision (FP16): {args.fp16}\n\n")
        
        f.write("## Training Results\n\n")
        f.write("| Epoch | Training Loss | Validation Loss |\n")
        f.write("|-------|--------------|----------------|\n")
        
        for stat in stats:
            f.write(f"| {stat['epoch']} | {stat['train_loss']:.4f} | {stat['val_loss']:.4f} |\n")
        
        f.write("\n## Best Model\n\n")
        best_epoch = min(stats, key=lambda x: x['val_loss'])
        f.write(f"- Epoch: {best_epoch['epoch']}\n")
        f.write(f"- Validation Loss: {best_epoch['val_loss']:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for healthcare")
    
    # Data arguments
    parser.add_argument("--data_dir", default="processed", help="Directory with processed data")
    parser.add_argument("--output_dir", default="models/finetuned", help="Output directory for model checkpoints")
    
    # Model arguments
    parser.add_argument("--base_model", default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for scheduler")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training info every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()