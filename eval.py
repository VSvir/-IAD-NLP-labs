import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from hqq.models.hf.base import AutoHQQHFModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# These parameters were obtained during experiments with baseline model (code in .ipynb file)
DEFAULT_BASELINE_METRIC = 0.7106
DEFAULT_ORIG_SIZE_MB = 15622.63
ORIG_PARAMS = 8.19
COMPRESSED_PARAMS = 4.72
BASELINE_MODEL_ID = "Qwen/Qwen3-8B"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_size_mb(path_or_id):
    """Count model size (for locally stored model only). 
    If HF ID used, returns default number (this number was obtained during experiments, 
    code is available in .ipynb file)."""
    if not os.path.exists(path_or_id):
        return 5893.87 
    
    total_size = 0
    for dirpath, _, filenames in os.walk(path_or_id):
        for f in filenames:
            if f.endswith(('.safetensors', '.bin', '.hqq_pack', '.pt')):
                total_size += os.path.getsize(os.path.join(dirpath, f))
    return total_size / (1024 * 1024)

def evaluate_mmlu_detailed(model, tokenizer, percentage=0.2):
    """Evaluate model performance on MMLU dataset. 
    Returns mean accuracy and accuracy by each subset in DataFrame format."""
    dataset = load_dataset("cais/mmlu", "all", split="test")
    df = dataset.to_pandas()
    data_df = df.groupby('subject', group_keys=False).apply(
        lambda x: x.sample(frac=percentage, random_state=42)
    )
    
    model.eval()
    choices = ['A', 'B', 'C', 'D']

    subjects = data_df['subject'].unique()
    subject_stats = {sub: {'correct': 0, 'total': 0} for sub in subjects}
    
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

def plot_mmlu_comparison(df_comp):
    categories = {
        'STEM': ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'statistics'],
        'Humanities': ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_dispute', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions'],
        'Social Sciences': ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'human_reproduction', 'public_relations', 'sociology', 'us_foreign_policy'],
        'Other': ['business_ethics', 'clinical_knowledge', 'global_facts', 'management', 'marketing', 'medical_genetics', 'nutrition', 'professional_accounting', 'professional_medicine', 'professional_psychology', 'virology']
    }
    def get_cat(sub):
        for c, subs in categories.items():
            if sub in subs: return c
        return 'Other'

    df_comp['category'] = df_comp['subject'].apply(get_cat)
    cat_plot = df_comp.groupby('category')[['baseline_acc', 'compressed_acc']].mean()
    
    cat_plot.plot(kind='bar', figsize=(10, 5))
    plt.title("MMLU: Baseline vs Compressed by Category")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    print("Plot saved as comparison_plot.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json', help="Path to config.json")
    parser.add_argument("--mode", type=str, choices=["csv_only", "run_quantized", "run_both"], default="run_quantized")
    parser.add_argument("--model_path", type=str, default="Neuro-Poplar/qwen3-8b-hqq-4bit")
    parser.add_argument("--csv_path", type=str, default="results_comparison.csv")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Load configs from file
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            for k, v in config.items(): setattr(args, k, v)

    global DEFAULT_BASELINE_METRIC
    global DEFAULT_ORIG_SIZE_MB
    global ORIG_PARAMS
    global COMPRESSED_PARAMS
    global BASELINE_MODEL_ID
    final_df = None
    b_acc = DEFAULT_BASELINE_METRIC
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # MODE 1: CSV only
    if args.mode == "csv_only":
        print(f"Loading results from {args.csv_path}...")
        final_df = pd.read_csv(args.csv_path, sep=';')
        # Expected columns: subject, baseline_acc, compressed_acc, count
        total_q = final_df['count'].sum()
        comp_acc = (final_df['compressed_acc'] * final_df['count']).sum() / total_q
        b_acc = (final_df['baseline_acc'] * final_df['count']).sum() / total_q

    # MODE 2: Infer quantized model 
    elif args.mode == "run_quantized":
        print(f"Loading Quantized Model: {args.model_path}...")
        model = AutoHQQHFModel.from_quantized(args.model_path, device=device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        COMPRESSED_PARAMS = sum(p.numel() for p in model.parameters()) / 1e9
        comp_acc, detailed_df = evaluate_mmlu_detailed(model, tokenizer, args.fraction)
        
        final_df = pd.read_csv(args.csv_path, sep=';')
        final_df['compressed_acc'] = detailed_df['accuracy']
        b_acc = (final_df['baseline_acc'] * final_df['count']).sum() / final_df['count'].sum()
        final_df.to_csv(args.csv_path, sep=';')

    # MODE 3: Run both models (baseline + quantized)
    elif args.mode == "run_both":
        # Baseline
        print("Evaluating Baseline Model...")
        b_model = AutoModelForCausalLM.from_pretrained(BASELINE_MODEL_ID, 
                                                       torch_dtype=torch.float16, 
                                                       device_map="auto")
        b_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_ID)
        ORIG_PARAMS = sum(p.numel() for p in b_model.parameters()) / 1e9
        b_acc, b_df = evaluate_mmlu_detailed(b_model, b_tokenizer, args.fraction)
        del b_model; torch.cuda.empty_cache()

        # Quantized
        print("Evaluating Compressed Model...")
        c_model = AutoHQQHFModel.from_quantized(args.model_path, device=device)
        c_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        COMPRESSED_PARAMS = sum(p.numel() for p in c_model.parameters()) / 1e9
        comp_acc, c_df = evaluate_mmlu_detailed(c_model, c_tokenizer, args.fraction)
        
        final_df = b_df.rename(columns={"accuracy": "baseline_acc"})
        final_df["compressed_acc"] = c_df["accuracy"]
        final_df.to_csv(args.csv_path, sep=';')

    ratio = DEFAULT_ORIG_SIZE_MB / get_size_mb(args.model_path)
    drop = max(0, (b_acc - comp_acc) / b_acc)
    score = ratio / (1 + drop)

    print("\n" + "="*30)
    print(f"Original parameters: {ORIG_PARAMS}B")
    print(f"Compressed parameters: {COMPRESSED_PARAMS}B")
    print(f"Original model size, MB: {DEFAULT_ORIG_SIZE_MB}")
    print(f"Compressed model size, MB: {get_size_mb(args.model_path)}")
    print(f"Ratio: {ratio:.4f}")
    print(f"Baseline Accuracy: {b_acc:.4f}")
    print(f"Compressed Accuracy: {comp_acc:.4f}")
    print(f"Performance Drop: {drop:.4f}")
    print(f"FINAL SCORE: {score:.4f}")
    print("="*30)

    if args.plot and final_df is not None:
        plot_mmlu_comparison(final_df)


if __name__ == "__main__":
    seed_everything(42)
    main()
