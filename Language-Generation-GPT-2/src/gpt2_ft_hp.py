import random

from exp_utils import create_exp_dir
import optuna
import argparse
import torch
import time
import math
import os
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from optimizer import create_adam_optimizer_from_args
from model import GPT2Config, GPT2LMModel
from data_utils import FT_Dataset
from gpt2_ft import train_validate, evaluate, compute_transformers_MergedLinear_grad_sample  # Ensure these functions are imported from gpt2_ft.py
from opacus import PrivacyEngine
from opacus.grad_sample import utils as opacus_utils
from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
import torch.distributed as dist

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

from loralib import MergedLinear
import loralib as lora

def get_args():
    """Returns a pre-defined set of hyperparameters, matching gpt2_ft.py."""
    args = argparse.Namespace(
        platform="local",
        local_rank=0,
        rank=0,
        device="cuda:0",
        world_size=1,
        random_seed=110,
        lr=0.0004,
        weight_decay=0.01,
        correct_bias=True,
        adam_epislon=1e-06,
        no_decay_bias=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        scheduler="constant",
        max_step=None,
        max_epoch=20,
        warmup_step=0,
        i_steps=0,
        i_lrs=0.00025,
        train_data="./data/e2e/train.jsonl",
        valid_data="./data/e2e/valid.jsonl",
        train_batch_size=1,
        valid_batch_size=1,
        grad_acc=1,
        clip=0.0,
        noise_multiplier=0.6,
        max_grad_norm=1.0,
        seq_len=512,
        model_card="gpt2.md",
        init_checkpoint="./pretrained_checkpoints/gpt2-medium-pytorch_model.bin",
        fp16=False,
        log_interval=100,
        eval_interval=2000,
        save_interval=1000,
        work_dir="./trained_models/GPT2_M/e2e",
        lora_dim=4,
        lora_alpha=32,
        obj="clm",
        lora_dropout=0.0,
        label_smooth=0.1,
        roll_interval=-1,
        roll_lr=1e-05,
        roll_step=100,
        eval_epoch=1
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    
    args.dist = dist 
    return args

def objective(trial):
    """Optuna objective function for hyperparameter tuning, closely resembling gpt2_ft.py."""
    args = get_args()
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    # Override specific parameters with Optuna suggestions
    # args.lora_dim = trial.suggest_int("lora_dim", 4, 128)
    # args.lora_alpha = trial.suggest_int("lora_alpha", 16, 256)
    # args.lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
    args.lr = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    # args.train_batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])
    # args.grad_acc = trial.suggest_int("grad_acc", 1, 4)
    
    
    # Load dataset
    train_data = FT_Dataset(args.train_data, args.train_batch_size, args.seq_len, joint_lm=args.obj=='jlm')
    valid_data = FT_Dataset(args.valid_data, args.valid_batch_size, args.seq_len)
    
    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
        
    )
    
    # Model configuration (assume medium)
    config = GPT2Config(
        n_embd=1024, n_layer=24, n_head=16,
        lora_attn_dim=args.lora_dim,
        lora_attn_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint))    

    lm_net = lm_net.cuda()
    
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    opacus_utils.register_grad_sampler(MergedLinear)(compute_transformers_MergedLinear_grad_sample)
    lm_net = DPDDP(lm_net)
    optimizer = create_adam_optimizer_from_args(lm_net, args)
    
    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)
    
        scheduler = create_optimizer_scheduler(optimizer, args)
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")

    n_layers = len([(n, p) for n, p in lm_net.named_parameters() if p.requires_grad])
    max_grad_norm = [args.max_grad_norm / np.sqrt(n_layers)] * n_layers

    ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    # We instead use the accountant from Gopi et al. (2021) as described in the paper.
    SAMPLE_RATE = (args.train_batch_size * args.grad_acc)/42061.0
    
    privacy_engine = PrivacyEngine(
        module=lm_net,
        sample_rate=SAMPLE_RATE, 
        alphas=ALPHAS,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)
    
    delta = 1.0/42061 # We instead use the accountant from Gopi et al. (2021) as described in the paper.

    train_step = 0
    best_val_loss = float("inf")
    for epoch in range(args.max_epoch):
        train_step = train_validate(
            lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
            train_step=train_step, epoch=epoch
        )
        
        # Evaluate
        val_loss, _ = evaluate(lm_net, valid_loader, args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

if __name__ == "__main__":
    args = get_args()
    
    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)) # Minimize validation loss
    study.optimize(objective, n_trials=20)
    
    print("Best hyperparameters:", study.best_params)
    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)