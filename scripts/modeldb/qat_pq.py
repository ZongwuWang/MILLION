import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)

import argparse
import json
import itertools
from tqdm import tqdm
import pathlib
import os
import importlib

from ..utils.Namespace import UniConfig, load_config
from ..utils.Timer import tprint, Timer
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import sys
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
    Timer('').start()
    # ================== Argument Parsing ==================
    parser = argparse.ArgumentParser(description="ModelDB")
    parser.add_argument("-f", "--file", type=str, help="Relative path to config.json. Relative to scripts/modeldb/configs/", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", required=True)
    parser.add_argument("-M", type=int, help="PQ config, number of sub-sections", required=True)
    parser.add_argument("--nbits", type=int, help="PQ config, number of bits per sub-section", required=True)
    parser.add_argument("--seed", type=int, help="Random seed", required=False, default=42)
    parser.add_argument("--save_dir", type=str, help="Directory to save the model", required=True)
    parser.add_argument("--deepspeed", type=str, help="Path to deepspeed config file", required=True)
    
    # Add local_rank argument (required for DeepSpeed)
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="Local rank passed from distributed launcher")

    args = parser.parse_args()
    
    # ================== Config ==================
    config = UniConfig()
    config.device = 'cuda'
    
    config.root = pathlib.Path(__file__).parent.parent.parent
    config.config_root = config.root / "scripts" / "modeldb" / "configs"
    config.config_path = config.config_root / args.file
    config.save_dir = pathlib.Path(args.save_dir)

    # Load config
    config += load_config(config.config_root / "default.json")
    config += load_config(config.config_path)

    if args.M is not None:
        config.M = args.M
    if args.nbits is not None:
        config.nbits = args.nbits
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.seed is not None:
        config.seed = args.seed

    config.model_root = config.root / "models"
    config.datasets_root = config.root / "datasets" 

    config.model_path = config.model_root / config.folder
    config.cent_root = config.root / "centroids" / config.model_name / config.dataset

    from transformers import AutoConfig
    from .models.ModelContext import get_context

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    config.context = get_context(config.model_config.model_type)

    # ================== Seed ==================
    seed_everything(config.seed)

    # ================== Load Model ==================
    tprint(f"Loading model {config.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import deepspeed
    
    with config.context.init_context:
        model = AutoModelForCausalLM.from_pretrained(config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        # model_engine, optimizer, _, _ = deepspeed.initialize(
        #     model=model,
        #     config=args.deepspeed,
        #     model_parameters=model.parameters()
        # )
            
    # ================== Initialize Codebook Register ==================
    tprint("Initializing codebook register")
    key_cent = torch.load(config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True).to(config.device)
    val_cent = torch.load(config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True).to(config.device)
    
    config.context.qat_codebook_register.init_register(model, key_cent, val_cent)
    
    # ================== Prepare Dataset ==================
    tprint(f"Preparing dataset {config.dataset}")
    from datasets import load_from_disk
    from transformers import DataCollatorForLanguageModeling

    raw_ds   = load_from_disk(str(config.datasets_root / config.dataset))
    train_ds = raw_ds["train"]
    val_ds   = raw_ds["validation"]
    
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_fn(ex):
        return tokenizer(ex["text"],
                        truncation=True,
                        max_length=4096,
                        padding=False)

    train_ds = train_ds.map(tokenize_fn,
                            batched=True,
                            remove_columns=["text"])
    val_ds   = val_ds.map(tokenize_fn,
                        batched=True,
                        remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # ================== Prepare Trainer ==================
    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(
        output_dir=config.save_dir,
        bf16=True,                      # Keep enabled for A100
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,   
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=3,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=200,                  # More frequent evaluation
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,     # Critical for memory
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        report_to="none",
        torch_compile=True              # Enable graph optimization
    )
    
    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
    )
    ## ================== QAT ===================
    tprint("Starting QAT")
    
    with config.context.qat_context, \
        config.context.qat_codebook_register:
            trainer.train()
            
    # ## ================== Save Model ===================
    # tprint("Saving model")
    # trainer.save_model(config.save_dir)
            
            