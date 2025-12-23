import random
from pathlib import Path

import numpy as np
import torch
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
SAVE_PATH = "./qwen3-8b-compressed"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compress():
    print(f"Loading model {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto"
    )

    hqq_config = BaseQuantizeConfig(nbits=4, group_size=128)

    print("Quantizing...")
    AutoHQQHFModel.quantize_model(
        model, 
        quant_config=hqq_config, 
        compute_dtype=torch.float16, 
        device="cuda"
    )

    print(f"Saving to {SAVE_PATH}...")
    AutoHQQHFModel.save_quantized(model, SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Compression finished successfully.\nModel saved to {Path(SAVE_PATH).resolve()}")


if __name__ == "__main__":
    seed_everything(42)
    compress()