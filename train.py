import torch
from datasets import load_dataset
from hqq.models.hf.base import AutoHQQHFModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

MAX_STEPS = 120 
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LEN = 256

def tokenize_function(examples):
    texts = [f"Question: {q}\nAnswer: {chr(65 + a)}" for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)

MODEL_ID = "Neuro-Poplar/qwen3-8b-hqq-4bit"
model = AutoHQQHFModel.from_quantized(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
dataset = dataset.shuffle(seed=42).select(range(1600))

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./qwen3-8b-hqq-4bit-finetuned",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=12,
    max_steps=MAX_STEPS,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained("./qwen3-8b-lora-adapter")
tokenizer.save_pretrained("./qwen3-8b-lora-adapter")

