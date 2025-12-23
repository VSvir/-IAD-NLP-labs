import argparse
import os
import pandas as pd
import torch
from datasets import load_dataset
from hqq.models.hf.base import AutoHQQHFModel
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

# Constants based on Phase 1 and Phase 2 results
ORIG_METRIC = 0.7106
ORIG_SIZE_MB = 15622.63
COMPRESSED_MODEL_SIZE_MB = 5893.87
COMPRESSED_METRIC = 0.6913
LORA_SIZE_FIXED = 29.28656768798828

def get_adapter_size_mb(path):
    """Calculate the size of LoRA adapter files in MB."""
    if not os.path.exists(path):
        return LORA_SIZE_FIXED
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(('.bin', '.safetensors', '.pt')):
                total_size += os.path.getsize(os.path.join(dirpath, f))
    return total_size / (1024 * 1024)

def evaluate_mmlu_detailed(model, tokenizer, percentage=0.2):
    """
    Evaluate model performance on MMLU dataset. 
    Returns micro-average accuracy and detailed results per subject.
    """
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df = dataset.to_pandas()
    # Stratified sampling by subject
    data_df = df.groupby('subject', group_keys=False).apply(
        lambda x: x.sample(frac=percentage, random_state=42)
    )
    
    model.eval()
    choices = ['A', 'B', 'C', 'D']
    subjects = data_df['subject'].unique()
    subject_stats = {sub: {'correct': 0, 'total': 0} for sub in subjects}
    
    # Get token IDs for A, B, C, D
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
    prompt_template = "Question: {question}\nChoices:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"

    with torch.no_grad():
        for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Evaluating"):
            prompt = prompt_template.format(
                question=row['question'],
                a=row['choices'][0], b=row['choices'][1],
                c=row['choices'][2], d=row['choices'][3]
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            
            # Get logits for the last token and extract choice probabilities
            last_logits = outputs.logits[0, -1, choice_ids]
            prediction = torch.argmax(last_logits).item()
            
            is_correct = (prediction == row['answer'])
            subject_stats[row['subject']]['total'] += 1
            if is_correct:
                subject_stats[row['subject']]['correct'] += 1

    results = []
    total_correct = 0
    total_samples = 0

    for sub, stats in subject_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        results.append({'subject': sub, 'accuracy': acc, 'count': stats['total']})
        total_correct += stats['correct']
        total_samples += stats['total']
        
    df_results = pd.DataFrame(results)
    mean_accuracy = total_correct / total_samples
                
    return mean_accuracy, df_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Neuro-Poplar/qwen3-8b-hqq-4bit")
    parser.add_argument("--adapter", type=str, default="Neuro-Poplar/qwen3-8b-hqq-4bit-lora-adapter")
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Loading Base HQQ Model: {args.base_model}...")
    # Load quantized base model
    base_model = AutoHQQHFModel.from_quantized(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Applying LoRA Adapter: {args.adapter}...")
    # Wrap base model with trained PEFT adapter
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    # 1. Calculate combined model size
    adapter_size = get_adapter_size_mb(args.adapter)
    total_compressed_size = COMPRESSED_MODEL_SIZE_MB + adapter_size
    ratio = ORIG_SIZE_MB / total_compressed_size

    # 2. Benchmark accuracy
    print("Starting MMLU evaluation...")
    tuned_metric, tuned_df = evaluate_mmlu_detailed(model, tokenizer, args.fraction)

    # 3. Calculate Final Score
    # Score = Compression_ratio / (1 + Performance_drop)
    drop = max(0, (ORIG_METRIC - tuned_metric) / ORIG_METRIC)
    score = ratio / (1 + drop)

    print("\n" + "="*40)
    print(f"Original Model Size: {ORIG_SIZE_MB:.2f} MB")
    print(f"Compressed Base Size: {COMPRESSED_MODEL_SIZE_MB:.2f} MB")
    print(f"LoRA Adapter Size: {adapter_size:.2f} MB")
    print(f"Total Combined Size: {total_compressed_size:.2f} MB")
    print(f"New Compression Ratio: {ratio:.4f}")
    print("-" * 40)
    print(f"Baseline Accuracy: {ORIG_METRIC:.4f}")
    print(f"Compressed Accuracy (Stage 1): {COMPRESSED_METRIC:.4f}")
    print(f"Fine-tuned Accuracy (Stage 2): {tuned_metric:.4f}")
    print(f"New Performance Drop: {drop:.4f}")
    print("-" * 40)
    print(f"FINAL SCORE (STAGE 2): {score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
    