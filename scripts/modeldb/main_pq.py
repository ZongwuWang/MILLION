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
    parser.add_argument("--weights", type=str, help="Path to model weights", required=False, default=None)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", required=False)
    parser.add_argument("-M", type=int, help="PQ config, number of sub-sections", required=False)
    parser.add_argument("--nbits", type=int, help="PQ config, number of bits per sub-section", required=False)
    parser.add_argument("-m", "--merged_training", action="store_true", help="Train a merged PQ", required=False)
    parser.add_argument("--opq", action="store_true", help="Prepand LT before PQ to maximize squred error across dimensions", required=False)
    parser.add_argument("--seed", type=int, help="Random seed", required=False, default=42)
    parser.add_argument("--half", action="store_true", help="Use half precision", required=False)
    parser.add_argument("--breakdown", action="store_true", help="Breakdown timing, could lead to overhead due to additional torch.cuda.synchronize", required=False)
    parser.add_argument(
        "-p", "--pipeline",
        nargs='+',
        choices=["baseline", "sampling", "training", "evaluation"],
        help="List of pipeline stages to execute"
    )

    args = parser.parse_args()

    if args.opq:
        raise NotImplementedError("OPQ is not implemented for GPU yet.")
    if args.merged_training is False:
        raise NotImplementedError("Only merged training is supported😈. Use --merged_training")
    
    # ================== Config ==================
    config = UniConfig()
    config.device = 'cuda' # TODO: support multi-gpu

    config.root = pathlib.Path(__file__).parent.parent.parent
    config.config_root = config.root / "scripts" / "modeldb" / "configs"
    config.config_path = config.config_root / args.file

    # Load config
    config += load_config(config.config_root / "default.json")
    config += load_config(config.config_path)

    if args.M is not None:
        config.M = args.M
    if args.nbits is not None:
        config.nbits = args.nbits
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.pipeline is not None:
        config.pipeline = args.pipeline
    if args.seed is not None:
        config.seed = args.seed
    if args.half is not None:
        config.half = args.half
    if args.breakdown is not None:
        config.breakdown = args.breakdown

    config.scalar_t = torch.float16 if config.half else torch.float32

    config.model_root = config.root / "models"
    config.datasets_root = config.root / "datasets" 

    if args.weights is not None:
        config.folder = pathlib.Path(args.weights).name
        config.model_path = pathlib.Path(args.weights)
    else:
        config.model_path = config.model_root / config.folder
    
    config.sample_root = config.root / "kv_samples" / config.model_name / config.dataset
    config.cent_root = config.root / "centroids" / config.model_name / config.dataset

    config.opq = args.opq
    config.merged_training = args.merged_training

    from transformers import AutoConfig
    from .models.ModelContext import get_context

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    config.context = get_context(config.model_config.model_type)

    from ..utils.pq_utils import nbits2dtype
    config.cache_dtype = nbits2dtype(config.nbits)

    # ================== Seed ==================
    seed_everything(config.seed)

    # ================== Load Model ==================
    if not (len(config.pipeline) == 1 and "training" in config.pipeline):
        tprint(f"Loading model {config.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with config.context.init_context:
            model = AutoModelForCausalLM.from_pretrained(config.model_path).to(config.device)
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            if config.half:
                model = model.half()

    # ================== baseline ==================
    if "baseline" in config.pipeline:
        tprint("Baseline")
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        with config.context.baseline_context:
            score_baseline = benchmark(model, tokenizer, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score_baseline, "model_name": config.model_name, "dataset": config.dataset, "baseline": True, "half": config.half}))
            f.write("\n")

    # ================== sampling ==================
    if "sampling" in config.pipeline and config.dataset != '_synthetic':
        tprint("Sampling")

        key_sampled_path = config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs'
        value_sampled_path = config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs'

        if key_sampled_path.exists() or value_sampled_path.exists():
            # ask for confirmation
            tprint(f"Sampling files already exist at {key_sampled_path} or {value_sampled_path}")
            tprint(f"Overwrite? (y/n)")
            if input().lower() != 'y':
                tprint("Exit")
                exit()
            else:
                os.remove(key_sampled_path)
                os.remove(value_sampled_path)

        from .Errors import SamplingComplete
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        config.sampled_nums = 0
        config.expected_sample_nums = 256 * 2**config.nbits
        config.expected_vecs_per_sample = config.model_config.num_key_value_heads
        config.threshold = 1 / config.expected_vecs_per_sample / 5

        with config.context.sampling_context:
            try:
                benchmark(model, tokenizer, **(config.to_dict()))
            except SamplingComplete as e:
                tprint(e)

    # ================== training ==================
    if "training" in config.pipeline and config.dataset != '_synthetic':
        tprint("Training")
        from ..utils.fvecio import read_fvecs
        from ..utils.pq_utils import train_pq
        from ..utils.pq_utils import train_opq
        from torch import save

        os.makedirs(config.cent_root, exist_ok=True)

        key = read_fvecs(config.sample_root / f'key_sampled_{config.M}_{config.nbits}.fvecs')
        if config.opq is False:
            key_cent = train_pq(key, config.M, config.nbits)
            save(key_cent, config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt')
            del key, key_cent
        else:
            key_A, key_cent = train_opq(key, config.M, config.nbits)
            save(key_A, config.cent_root / f'key_A_{config.M}_{config.nbits}.opq.pt')
            save(key_cent, config.cent_root / f'key_cent_{config.M}_{config.nbits}.opq.pt')
            del key, key_A, key_cent

        val = read_fvecs(config.sample_root / f'value_sampled_{config.M}_{config.nbits}.fvecs')
        if config.opq is False:
            val_cent = train_pq(val, config.M, config.nbits)
            save(val_cent, config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt')
            del val, val_cent
        else:
            val_A, val_cent = train_opq(val, config.M, config.nbits)
            save(val_A, config.cent_root / f'val_A_{config.M}_{config.nbits}.opq.pt')
            save(val_cent, config.cent_root / f'val_cent_{config.M}_{config.nbits}.opq.pt')
            del val, val_A, val_cent

    if "evaluation" not in config.pipeline:
        tprint("Exit")
        exit()

    # ================== PQ Config ==================
    if "evaluation" in config.pipeline:
        if config.dataset == '_synthetic':
            tprint("Using synthetic centroids for speed evaluation")
            key_cent = torch.randn(config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
            val_cent = torch.randn(config.M, 2**config.nbits, config.d // config.M, dtype=model.dtype, device=config.device)
        else:
            key_cent = torch.load(config.cent_root / f'key_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True)
            key_cent = key_cent.to(config.device).to(model.dtype)
            val_cent = torch.load(config.cent_root / f'val_cent_{config.M}_{config.nbits}.pq.pt', weights_only=True)
            val_cent = val_cent.to(config.device).to(model.dtype)

            if config.opq is True:
                tprint("OPQ patching model weights")
                key_A = torch.load(config.cent_root / f'key_A_{config.M}_{config.nbits}.opq.pt', weights_only=True)
                key_A = key_A.to(config.device).to(model.dtype)
                val_A = torch.load(config.cent_root / f'val_A_{config.M}_{config.nbits}.opq.pt', weights_only=True)
                val_A = val_A.to(config.device).to(model.dtype)

                # # OPQ prepends Linear Transformation before PQ: 
                # # key_codes = opq_encode(K) = pq_encode(K @ A.T) = pq_encode(X @ Wk.T @ A.T) = pq_encode(X @ (A @ Wk).T)

                # # Insight: A @ Wk is a fixed transformation, so we can precompute it as new model weights to avoid online overhead

                # # In practice: k_proj, v_proj are instances of nn.Linear
                # # k_proj = x -> x @ Wk.T where k_proj.weight = Wk, sized (num_key_value_heads * head_dim, hidden_size)
                # # We are assuming bias is not used in k_proj, v_proj. Fortunately this is generally true for most models.
                # # Merged training uses the same opq for all heads, A.shape = (head_dim, head_dim). To apply A to all heads, 
                # # we should fit A to diagonal blocks of (num_key_value_heads * head_dim, num_key_value_heads * head_dim) matrix

                # head_dim = config.model_config.hidden_size // config.model_config.num_key_value_heads
                # head_dim_sum = head_dim * config.model_config.num_key_value_heads
                # key_A_expanded = torch.zeros(head_dim_sum, head_dim_sum, dtype=model.dtype, device=model.device)
                # val_A_expanded = torch.zeros(head_dim_sum, head_dim_sum, dtype=model.dtype, device=model.device)
                # for i in range(config.model_config.num_key_value_heads):
                #     start = i * head_dim
                #     end = start + head_dim
                #     key_A_expanded[start:end, start:end] = key_A
                #     val_A_expanded[start:end, start:end] = val_A

                # # The following code is tuned for llama. For other models, we may need to change the component names.
                # for layer in model.model.layers:
                #     Wk = layer.self_attn.k_proj.weight.data
                #     Wv = layer.self_attn.v_proj.weight.data
                #     layer.self_attn.k_proj.weight = torch.nn.Parameter(key_A_expanded @ Wk)
                #     layer.self_attn.v_proj.weight = torch.nn.Parameter(val_A_expanded @ Wv)
                # del key_A_expanded, val_A_expanded

                del key_A, val_A, 
                torch.cuda.empty_cache()

        from ..utils.pq_utils import DynamicPQCache
        cache = DynamicPQCache(
            bs = 1, # TODO: support batch size?
            num_key_value_heads=config.model_config.num_key_value_heads,
            nh = config.model_config.num_attention_heads,
            M = config.M,
            layer_num=config.model_config.num_hidden_layers,
            dtype=config.cache_dtype,
            nbits=config.nbits,
            d=config.d,
            scalar_t=config.scalar_t,
        )
        cache.set_cent(key_cent, val_cent)
    
        tprint("Evaluation")
        
        from ..benchmarks import dataset2benchmark
        benchmark = dataset2benchmark[config.dataset]

        with config.context.evaluation_context:
            score = benchmark(model, tokenizer, cache_clear_func=cache.init_cache, **(config.to_dict()))

        # write to jsonl
        with open(config.root / "scripts" / "modeldb" / "results.jsonl", "a") as f:
            f.write(json.dumps({"score": score, **config.to_serializable_dict()}))
            f.write("\n")

    tprint("Exit")
