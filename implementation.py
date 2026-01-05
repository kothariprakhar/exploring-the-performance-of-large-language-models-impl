import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
import numpy as np
from typing import List, Dict
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------------------------------------------
# 1. Dummy Data: Subjective Span Identification Tasks
# ------------------------------------------------------------------
# The paper focuses on tasks like Sentiment Analysis (aspects), 
# Offensive Language, and Claim Verification.

raw_data = [
    {
        "text": "The restaurant ambiance was cozy, but the service was incredibly slow.",
        "task": "sentiment",
        "label": "negative",
        "gold_span": "service was incredibly slow"
    },
    {
        "text": "I think that statement is absolute garbage and you should feel bad.",
        "task": "offensive",
        "label": "insult",
        "gold_span": "absolute garbage"
    },
    {
        "text": "The study claims that drinking water cures all diseases instantly.",
        "task": "claim_verification",
        "label": "false",
        "gold_span": "cures all diseases instantly"
    },
    {
        "text": "I love the battery life on this phone, it lasts for two days.",
        "task": "sentiment",
        "label": "positive",
        "gold_span": "lasts for two days"
    }
]

# ------------------------------------------------------------------
# 2. Prompt Engineering Strategies (ICL, CoT)
# ------------------------------------------------------------------

class PromptStrategy:
    @staticmethod
    def zero_shot(text, label, task):
        """Standard Instruction Prompting."""
        return (
            f"Task: Identify the text span in the input that supports the label '{label}' for {task}.\n"
            f"Input: {text}\n"
            f"Span:"
        )

    @staticmethod
    def few_shot(text, label, task, examples):
        """In-Context Learning (ICL)."""
        prompt = f"Task: Identify the text span supporting the label for {task}.\n\n"
        for ex in examples:
            prompt += f"Input: {ex['text']}\nLabel: {ex['label']}\nSpan: {ex['gold_span']}\n\n"
        
        prompt += f"Input: {text}\nLabel: {label}\nSpan:"
        return prompt

    @staticmethod
    def chain_of_thought(text, label, task):
        """Chain of Thought (CoT) - Asking the model to reason first."""
        return (
            f"Task: Identify the text span in the input that supports the label '{label}' for {task}.\n"
            f"Input: {text}\n"
            f"Let's think step by step to find the reason for the label, then extract the exact substring.\n"
            f"Reasoning and Span:"
        )

# ------------------------------------------------------------------
# 3. Dataset & Model Setup
# ------------------------------------------------------------------

class SubjectiveSpanDataset(Dataset):
    def __init__(self, data, tokenizer, strategy="zero_shot", examples=None, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.examples = examples if examples else []
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Select Prompt Strategy
        if self.strategy == "few_shot":
            # Pick random examples excluding current one to avoid leakage
            context_examples = [x for i, x in enumerate(self.examples) if x['text'] != item['text']]
            context_examples = context_examples[:2] # 2-shot
            input_text = PromptStrategy.few_shot(item['text'], item['label'], item['task'], context_examples)
        elif self.strategy == "cot":
            input_text = PromptStrategy.chain_of_thought(item['text'], item['label'], item['task'])
        else: # zero_shot / instruction_tuning
            input_text = PromptStrategy.zero_shot(item['text'], item['label'], item['task'])

        target_text = item['gold_span']

        # Tokenize
        source = self.tokenizer(
            input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        target = self.tokenizer(
            target_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze(),
            "raw_text": item['text'],
            "gold_span": item['gold_span']
        }

# ------------------------------------------------------------------
# 4. Evaluation Metric (Token Overlap / F1)
# ------------------------------------------------------------------

def calculate_f1(pred, gold):
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    
    if len(pred_toks) == 0:
        return 0.0
    
    common = pred_toks.intersection(gold_toks)
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# ------------------------------------------------------------------
# 5. Main Execution Flow
# ------------------------------------------------------------------

def main():
    # Use FLAN-T5-Small as a proxy for larger LLMs (T5 is instruction tuned)
    # It runs easily on CPU for demonstration.
    model_name = "google/flan-t5-small"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- EXPERIMENT 1: Zero-Shot Inference (Base Model Performance) ---
    print("\n--- Experiment 1: Zero-Shot Inference ---")
    dataset = SubjectiveSpanDataset(raw_data, tokenizer, strategy="zero_shot")
    loader = DataLoader(dataset, batch_size=1)
    
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        outputs = model.generate(input_ids, max_length=50)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Original: {batch['raw_text'][0]}")
        print(f"Gold Span: {batch['gold_span'][0]}")
        print(f"Predicted: {pred_text}")
        print(f"F1 Score: {calculate_f1(pred_text, batch['gold_span'][0]):.2f}\n")

    # --- EXPERIMENT 2: Instruction Tuning (Simulating Fine-Tuning) ---
    # The paper discusses fine-tuning LLMs on the specific task.
    print("--- Experiment 2: Instruction Tuning (Fine-Tuning) ---")
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Use the same dataset configuration for training
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # HF models handle label shifting internally
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # --- EXPERIMENT 3: Evaluation after Tuning ---
    print("\n--- Experiment 3: Evaluation After Instruction Tuning ---")
    model.eval()
    for batch in loader: # Using the same loader for simplicity (in real research, use test set)
        input_ids = batch["input_ids"].to(device)
        outputs = model.generate(input_ids, max_length=50)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Predicted: {pred_text} | Gold: {batch['gold_span'][0]}")

    # --- EXPERIMENT 4: Few-Shot In-Context Learning (Demonstration) ---
    print("\n--- Experiment 4: Few-Shot In-Context Learning Strategy ---")
    # Note: We use the raw untuned model logic conceptually here, 
    # but we are running on the now-tuned model for code simplicity.
    fs_dataset = SubjectiveSpanDataset(raw_data, tokenizer, strategy="few_shot", examples=raw_data)
    fs_loader = DataLoader(fs_dataset, batch_size=1)
    
    batch = next(iter(fs_loader))
    # Just printing the prompt to show the strategy implementation
    decoded_prompt = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
    print("Constructed Few-Shot Prompt (truncated for display):")
    print(decoded_prompt[:300] + "...")

if __name__ == "__main__":
    main()
