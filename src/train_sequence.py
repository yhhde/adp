"""
Multilingual Text Classification for Alzheimer Detection via Traditional Sequence Classification.
Unified Implementation with Native02 Infrastructure and Run_Classify Training Control.
================================================================================================
This script combines:
- run_classify.py: Manual training loop with full control and distributed training support
- finetune_sequence.py: AD-specialized SequenceDataset and memory optimization  
- native02.py: Unified experiment management and infrastructure

Key Features:
- Traditional [CLS] token classification
- Manual training loop for maximum control
- Unified experiment management (native02 style)
- Optional distributed training support
- 16GB GPU memory optimization
- No TensorBoard (simplified monitoring)
- No optimizer/scheduler state saving (simplified checkpoints)
"""

import torch
import torch.nn.functional as F
import argparse
import json
import os
import logging
import datetime
import csv
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from collections import Counter
from tqdm import tqdm, trange

from datasets import Dataset, DatasetDict, load_dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support

# Import unified utilities (native02 standard)
from common_utils import (
    ExperimentConfig, set_random_seed, create_unified_experiment_dirs,
    create_unified_external_eval_dirs, create_unified_prediction_dirs,
    analyze_dataset_statistics, print_dataset_statistics, save_dataset_statistics,
    filter_dataset_by_language, save_predictions_csv, 
    save_model_config, load_model_config, create_unified_experiment_hash,
    # Unified logging functions
    setup_unified_logging, log_epoch_progress_unified, generate_training_summary_unified,
    # Unified registration functions
    register_experiment_unified, register_best_model_unified, create_experiment_summary_unified,
    # Unified checkpoint function (simplified usage)
    save_checkpoint_unified,
    # Per-class metrics function
    compute_per_class_metrics
)

# Optional distributed training support
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
try:
    from torch.utils.data.distributed import DistributedSampler
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Set environment variable to suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SequenceDataset(torch.utils.data.Dataset):
    """Traditional sequence classification dataset using [CLS] token (from sequence.py)."""
    
    def __init__(self, examples, tokenizer, max_length, label2id):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        label = example['label']
        
        # Standard tokenization for sequence classification
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert string label to integer
        class_label = self.label2id[label]
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(class_label, dtype=torch.long)
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in tokenized:
            result['token_type_ids'] = tokenized['token_type_ids'].squeeze()
        
        return result


def compute_sequence_loss(model, batch, class_weights=None, device='cuda'):
    """Compute traditional sequence classification loss using [CLS] token (from sequence.py)."""
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Prepare model inputs
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    # Add token_type_ids if available
    if 'token_type_ids' in batch:
        model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)
    
    # Forward pass
    outputs = model(**model_inputs)
    
    # Extract loss and logits
    if class_weights is not None:
        # Custom loss with class weights
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
    else:
        # Use model's built-in loss
        loss = outputs.loss
        logits = outputs.logits
    
    return loss, logits


def setup_distributed_if_available(args):
    """
    NEW FUNCTION: Auto-detect and setup distributed training if available.
    Does not modify existing utils functions.
    """
    if not DISTRIBUTED_AVAILABLE:
        args.local_rank = -1
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return args
    
    if args.local_rank == -1:
        # Single GPU or CPU training
        device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.force_cpu else 0
    else:
        # Distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    
    args.device = device
    return args


def create_distributed_dataloader(dataset, batch_size, shuffle=False, args=None):
    """
    NEW FUNCTION: Create appropriate dataloader for single/distributed training.
    """
    if args and args.local_rank != -1 and DISTRIBUTED_AVAILABLE:
        # Distributed training
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling
    else:
        # Single GPU training
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        shuffle = False  # When using a custom sampler, shuffle must be False
    
    return DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,  # Memory optimization for 16GB GPU
        drop_last=False
    )


def train_epoch_unified(model, dataloader, optimizer, scheduler, args, 
                       class_weights, device, scaler=None, logger=None):
    """
    Unified training epoch combining run_classify control + sequence memory optimization.
    Inspired by run_classify.py manual training loop but simplified (no TensorBoard).
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    # Progress bar (similar to run_classify style)
    epoch_iterator = tqdm(dataloader, desc="Training", 
                         disable=args.local_rank not in [-1, 0],
                         bar_format='{l_bar}{bar:20}{r_bar}')
    
    for step, batch in enumerate(epoch_iterator):
        # Compute loss using sequence classification
        if scaler is not None:  # Mixed precision
            with torch.amp.autocast('cuda'):
                loss, _ = compute_sequence_loss(model, batch, class_weights, device)
        else:  # Regular precision
            loss, _ = compute_sequence_loss(model, batch, class_weights, device)
        
        # Gradient accumulation (run_classify style)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item()
        
        # Optimizer step with gradient accumulation
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == num_batches:
            # Gradient clipping (run_classify style)
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Memory optimization (sequence.py style) - for 16GB GPU
        if (step + 1) % (args.gradient_accumulation_steps * 4) == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar less frequently
        if step % 10 == 0:
            epoch_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Clean up batch to free memory
        del batch, loss
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    
    return total_loss / num_batches


def evaluate_epoch_unified(model, dataloader, class_weights, device, args, scaler=None):
    """Unified evaluation epoch (simplified from run_classify + sequence)."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        eval_iterator = tqdm(dataloader, desc="Evaluating", 
                           disable=args.local_rank not in [-1, 0],
                           bar_format='{l_bar}{bar:20}{r_bar}')
        
        for step, batch in enumerate(eval_iterator):
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss, logits = compute_sequence_loss(model, batch, class_weights, device)
            else:
                loss, logits = compute_sequence_loss(model, batch, class_weights, device)
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Clean up
            del batch, loss, logits, predictions, labels
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions, average='macro')
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(dataloader)
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return avg_loss, f1, accuracy, all_predictions, all_labels


def setup_optimizer_scheduler(model, train_dataloader, args):
    """Setup optimizer and learning rate scheduler (from sequence.py + run_classify.py)."""
    # Calculate total steps
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Setup optimizer (run_classify style with weight decay groups)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    
    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler, total_steps, warmup_steps


def predict_dataset_unified(model, dataset, device, batch_size=1, args=None):
    """Unified prediction function for final evaluation."""
    model.eval()
    dataloader = create_distributed_dataloader(dataset, batch_size, shuffle=False, args=args)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", disable=args.local_rank not in [-1, 0]):
            _, logits = compute_sequence_loss(model, batch, device=device)
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            del batch, logits, predictions, labels
    
    torch.cuda.empty_cache()
    return all_predictions, all_labels


def main():
    # Setup arguments (combining run_classify + sequence + native02)
    parser = argparse.ArgumentParser(description='Unified Traditional Sequence Classification for Alzheimer Detection')
    
    # Essential parameters (native02 style)
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data folder.')
    parser.add_argument('--output_dir', type=str, default='experiments/', help='Base path for outputs.')
    parser.add_argument("--result_dir", type=str, default='experiments/', help="Path for result files.")
    parser.add_argument("--log_path", type=str, default='experiments/', help="Path for log files.")
    
    parser.add_argument('--model_id', type=str, default='bert-base-multilingual-uncased', help='Model name.')
    
    # Checkpoint and evaluation parameters (native02 style)
    parser.add_argument('--checkpoint_id', type=str, default=None, help='Path to model checkpoint for prediction.')
    parser.add_argument('--external_eval_name', type=str, default=None, help='Name for external evaluation dataset.')
    
    # Language filtering (native02 style)
    parser.add_argument('--languages', type=str, nargs='+', default=None, help='Languages to use.')
    parser.add_argument('--english_only', action='store_true', help='Use English only.')
    
    # Training parameters (optimized for 16GB GPU)
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size (16GB GPU optimized).')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation (effective batch=32).')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs.')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length.')
    
    # Advanced training parameters (run_classify style)
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm.')
    
    # Utility parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--class_weights', action='store_true', help='Use class weights.')
    parser.add_argument('--predict_only', action='store_true', help='Only run prediction.')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage.')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training (recommended for 16GB GPU).')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    
    # Distributed training parameters (run_classify style, optional)
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training.')
    
    # Early stopping parameters (run_classify style)
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping.')
    parser.add_argument('--tolerance', type=int, default=3, help='Early stopping tolerance.')
    
    args = parser.parse_args()
    
    # Setup distributed training if available
    args = setup_distributed_if_available(args)
    device = args.device
    
    # Initialize config
    config = ExperimentConfig()
    args.method = 'sequence'
    
    # Load model configuration if using checkpoint
    model_config = None
    if args.predict_only and args.checkpoint_id:
        try:
            model_config = load_model_config(args.checkpoint_id)
            print(f"📋 Loaded model configuration from {args.checkpoint_id}")
            
            # Override args with saved configuration for consistency
            if 'experiment_args' in model_config:
                saved_args = model_config['experiment_args']
                args.model_id = saved_args.get('model_id', args.model_id)
                args.max_length = saved_args.get('max_length', args.max_length)
        except Exception as e:
            print(f"⚠️ Could not load model config: {e}, using current args")
    
    # Print unified architecture information
    if args.local_rank in [-1, 0]:
        print(f"\n🔧 UNIFIED SEQUENCE CLASSIFICATION ARCHITECTURE:")
        print(f"   Training Control: run_classify.py manual training loop")
        print(f"   AD Specialization: sequence.py SequenceDataset and loss computation")
        print(f"   Infrastructure: native02.py unified experiment management")
        print(f"   Model: AutoModelForSequenceClassification")
        print(f"   Classification: [CLS] token + linear layer")
        print(f"   Memory Optimization: 16GB GPU friendly (batch_size={args.train_batch_size}, accumulation={args.gradient_accumulation_steps})")
        print(f"   Distributed: {'Enabled' if args.local_rank != -1 else 'Single GPU'}")
        print(f"   Mixed Precision: {'Enabled' if args.fp16 else 'Disabled'}")
    
    # Handle language filtering
    if args.english_only:
        args.languages = ['en']
    
    # Create unified directories (native02 style)
    if not args.predict_only:
        # Training: create unified directories with experiment hash
        base_dirs = {'output': args.output_dir, 'result': args.result_dir, 'log': args.log_path}
        try:
            dirs, timestamp, experiment_name, experiment_hash = create_unified_experiment_dirs(
                base_dirs, args, method="sequence", config=config
            )
            if args.local_rank in [-1, 0]:
                print(f"🚀 Experiment: {experiment_name}")
                print(f"🔒 Isolation Hash: {experiment_hash}")
                print(f"📁 Unified Directory: {dirs['output']}")
        except Exception as e:
            if args.local_rank in [-1, 0]:
                print(f"❌ Error creating directories: {e}")
            return
    else:
        # Prediction only: create unified directory structure for external evaluation
        experiment_hash = "prediction"
        base_dirs = {'output': args.output_dir, 'result': args.result_dir, 'log': args.log_path}
        
        if args.external_eval_name:
            # External dataset evaluation
            dirs, timestamp, experiment_name = create_unified_external_eval_dirs(
                base_dirs, args, args.external_eval_name, method="sequence", config=config
            )
            if args.local_rank in [-1, 0]:
                model_short = config.model_name_map.get(args.model_id, args.model_id.replace('/', '-'))
                print(f"🔍 External Evaluation: {args.external_eval_name}")
                print(f"🎯 Model: {model_short} Sequence Classification")
        else:
            # Regular prediction
            dirs, timestamp, experiment_name = create_unified_prediction_dirs(
                base_dirs, args, method="sequence", config=config
            )
            if args.local_rank in [-1, 0]:
                print(f"🔍 Prediction: {experiment_name}")
        
        if args.local_rank in [-1, 0]:
            print(f"📁 Using checkpoint: {args.checkpoint_id}")
            print(f"📂 Output directory: {dirs['output']}")
    
    # Setup unified logging (native02 style)
    if args.local_rank in [-1, 0]:
        log_file = os.path.join(dirs['log'], f"{experiment_name}.log")
        logger = setup_unified_logging(log_file, experiment_name, args.verbose)
        
        logger.info(f"🔧 UNIFIED SEQUENCE CLASSIFICATION EXPERIMENT")
        logger.info(f"Device: {device}")
        logger.info(f"N_GPU: {args.n_gpu}")
        logger.info(f"Distributed: {args.local_rank != -1}")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Method: Traditional [CLS] token classification")
        logger.info(f"Architecture: AutoModelForSequenceClassification")
        logger.info(f"Mixed Precision: {args.fp16}")
        if args.predict_only:
            logger.info(f"Checkpoint: {args.checkpoint_id}")
        else:
            logger.info(f"Experiment Hash: {experiment_hash}")
            logger.info(f"Batch Size: {args.train_batch_size}")
            logger.info(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
            logger.info(f"Effective Batch Size: {args.train_batch_size * args.gradient_accumulation_steps * max(1, args.n_gpu)}")
    else:
        logger = None
    
    # Set deterministic environment (native02 style)
    set_random_seed(args.seed)
    if logger:
        logger.info(f"🌱 Random seed set to: {args.seed}")
    
    # Load dataset (native02 style)
    data_files = {
        'train': os.path.join(args.data_path, 'train.csv'),
        'val': os.path.join(args.data_path, 'val.csv'),
        'test': os.path.join(args.data_path, 'test.csv'),
    }
    
    if logger:
        logger.info("📊 Loading dataset...")
    dataset = load_dataset('csv', data_files=data_files)
    
    # Analyze dataset (native02 style)
    original_stats = analyze_dataset_statistics(dataset)
    if args.verbose and args.local_rank in [-1, 0]:
        print_dataset_statistics(original_stats)
    if args.local_rank in [-1, 0]:
        save_dataset_statistics(original_stats, dirs['result'], experiment_name)
    
    # Load tokenizer and model
    if logger:
        logger.info(f"🔤 Loading tokenizer and model...")
    try:
        # Load tokenizer from checkpoint if available, otherwise from model_id
        tokenizer_path = args.checkpoint_id if args.predict_only and args.checkpoint_id else args.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load tokenizer: {e}. Trying with add_prefix_space=True")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)
    
    # Filter dataset by language if specified (native02 style)
    if args.languages:
        if logger:
            logger.info(f"🌍 Filtering dataset for languages: {args.languages}")
        dataset = filter_dataset_by_language(dataset, args.languages)
        if dataset is None or len(dataset) == 0:
            if logger:
                logger.error("❌ No data found for specified languages!")
            return
        
        filtered_stats = analyze_dataset_statistics(dataset)
        if args.verbose and args.local_rank in [-1, 0]:
            print("\nFILTERED DATASET:")
            print_dataset_statistics(filtered_stats)
        if args.local_rank in [-1, 0]:
            save_dataset_statistics(filtered_stats, dirs['result'], f'{experiment_name}_filtered')
    
    # Calculate class weights if needed (skip for prediction only)
    class_weights = None
    if args.class_weights and not args.predict_only:
        train_labels = [item['label'] for item in dataset['train']]
        train_label_ids = [config.label2id[label] for label in train_labels]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_label_ids),
            y=train_label_ids
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
        if logger:
            logger.info(f"⚖️ Class weights: {dict(zip(config.id2label.values(), class_weights.cpu().tolist()))}")
    
    # Load model
    if args.predict_only and args.checkpoint_id:
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_id).to(device)
        if logger:
            logger.info(f"📁 Loaded model from checkpoint: {args.checkpoint_id}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id, 
            num_labels=config.num_labels, 
            trust_remote_code=True
        ).to(device)
        if logger:
            logger.info(f"🤖 Loaded model: {args.model_id}")
    
    # Setup distributed model if needed
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    
    if logger:
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sequence classification datasets
    if logger:
        logger.info("🔄 Creating sequence classification datasets...")
    train_dataset = SequenceDataset(
        dataset['train'], tokenizer, args.max_length, config.label2id
    )
    val_dataset = SequenceDataset(
        dataset['val'], tokenizer, args.max_length, config.label2id
    )
    test_dataset = SequenceDataset(
        dataset['test'], tokenizer, args.max_length, config.label2id
    )
    
    # Create distributed dataloaders
    train_dataloader = create_distributed_dataloader(
        train_dataset, args.train_batch_size, shuffle=True, args=args
    )
    val_dataloader = create_distributed_dataloader(
        val_dataset, args.eval_batch_size, shuffle=False, args=args
    )
    test_dataloader = create_distributed_dataloader(
        test_dataset, args.eval_batch_size, shuffle=False, args=args
    )
    
    if logger:
        logger.info(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Setup optimizer and scheduler (skip for prediction only)
    if not args.predict_only:
        optimizer, scheduler, total_steps, warmup_steps = setup_optimizer_scheduler(
            model, train_dataloader, args
        )
        
        if logger:
            logger.info(f"🔧 Training setup:")
            logger.info(f"  Total steps: {total_steps}")
            logger.info(f"  Warmup steps: {warmup_steps}")
            logger.info(f"  Steps per epoch: {len(train_dataloader) // args.gradient_accumulation_steps}")
        
        # Setup mixed precision if requested (16GB GPU optimization)
        scaler = torch.amp.GradScaler('cuda') if args.fp16 and device.type == 'cuda' else None
        if scaler and logger:
            logger.info("🚀 Mixed precision training enabled (16GB GPU optimization)")
    
    # Save model configuration (native02 style)
    model_config = {
        'method': 'sequence',
        'model_id': args.model_id,
        'tokenizer_config': {
            'max_length': args.max_length,
            'padding': 'max_length',
            'truncation': True
        },
        'label_mappings': {
            'label2id': config.label2id,
            'id2label': config.id2label,
            'num_labels': config.num_labels
        },
        'experiment_args': vars(args),
        'experiment_hash': experiment_hash if 'experiment_hash' in locals() else 'unknown'
    }
    
    # Training loop (run_classify style manual control)
    if not args.predict_only:
        if logger:
            logger.info("🚀 Starting unified sequence classification training...")
        
        best_f1 = 0.0
        patience_counter = 0
        checkpoint_dir = os.path.join(dirs['output'], "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_val_f1 = 0.0
        
        # Training iterator (run_classify style)
        train_iterator = trange(
            int(args.num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
        
        for epoch in train_iterator:
            if logger:
                logger.debug(f"🔄 Starting Epoch {epoch + 1}/{args.num_epochs}")
            
            # Training phase
            train_loss = train_epoch_unified(
                model, train_dataloader, optimizer, scheduler, args,
                class_weights, device, scaler, logger
            )
            
            # Validation phase
            val_loss, val_f1, val_accuracy, _, _ = evaluate_epoch_unified(
                model, val_dataloader, class_weights, device, args, scaler
            )
            
            final_val_f1 = val_f1
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            
            # Check if this is the best model
            is_best = val_f1 > best_f1
            
            # Log epoch progress using unified logging (native02 style)
            if logger:
                log_epoch_progress_unified(
                    logger, epoch, args.num_epochs, train_loss, val_loss, val_f1,
                    best_f1, patience_counter, args.early_stopping_patience, 
                    current_lr, is_best, style="simple"
                )
            
            # Early stopping and model saving (run_classify style logic)
            if is_best:
                best_f1 = val_f1
                patience_counter = 0
                
                if logger:
                    logger.debug(f"💾 New best F1: {best_f1:.6f}")
            else:
                patience_counter += 1
                if args.early_stopping and patience_counter >= args.tolerance:
                    if logger:
                        logger.info(f"⏹️ Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save checkpoint using unified function (with full optimizer/scheduler state)
            if args.local_rank in [-1, 0]:
                save_checkpoint_unified(
                    model=model.module if hasattr(model, 'module') else model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,      # Use real optimizer object
                    scheduler=scheduler,      # Use real scheduler object
                    epoch=epoch,
                    best_f1=best_f1,
                    checkpoint_dir=checkpoint_dir,
                    experiment_name=experiment_name,
                    experiment_hash=experiment_hash,
                    is_best=is_best,
                    method="sequence",
                    logger=logger
                )
        
        if logger:
            logger.info("✅ Training completed!")
    
    # Final evaluation (native02 style)
    if args.local_rank in [-1, 0]:
        if logger:
            logger.info("🔍 FINAL EVALUATION")
        
        def evaluate_and_save_unified(split_name, dataset_split, original_split):
            """Unified evaluation and save function (native02 style)."""
            if logger:
                logger.debug(f"📊 Evaluating on {split_name} set...")
            
            predictions, true_labels = predict_dataset_unified(
                model.module if hasattr(model, 'module') else model, 
                dataset_split, device, args.eval_batch_size, args
            )
            
            # Calculate overall metrics
            f1 = f1_score(true_labels, predictions, average='macro')
            accuracy = accuracy_score(true_labels, predictions)
            
            # Calculate detailed per-class metrics using sklearn
            precision, recall, f1_per_class, support = precision_recall_fscore_support(
                true_labels, predictions, average=None, labels=list(range(config.num_labels))
            )
            
            # Calculate macro averages
            macro_precision = precision.mean()
            macro_recall = recall.mean()
            macro_f1 = f1_per_class.mean()
            
            # Generate classification report
            class_names = [config.id2label[i] for i in range(config.num_labels)]
            class_report = classification_report(
                true_labels, predictions, 
                target_names=class_names, 
                digits=4
            )
            
            # Calculate confusion matrix
            confusion_matrix = np.zeros((config.num_labels, config.num_labels), dtype=int)
            for p, l in zip(predictions, true_labels):
                confusion_matrix[l][p] += 1
            
            # Use existing compute_per_class_metrics function for consistency
            per_class_metrics = compute_per_class_metrics(confusion_matrix, config.id2label)
            
            # ============= DETAILED CONSOLE OUTPUT =============
            if logger:
                logger.info(f"\n" + "="*60)
                logger.info(f"📊 DETAILED {split_name.upper()} RESULTS")
                logger.info(f"="*60)
                
                # Overall metrics
                logger.info(f"🎯 Overall Metrics:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Macro F1: {macro_f1:.4f}")
                logger.info(f"  Macro Precision: {macro_precision:.4f}")
                logger.info(f"  Macro Recall: {macro_recall:.4f}")
                
                # Per-class metrics (detailed display)
                logger.info(f"\n🏥 Per-Class Disease Metrics:")
                for i, class_name in enumerate(class_names):
                    logger.info(f"  {class_name:>5} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1: {f1_per_class[i]:.4f} | Support: {support[i]}")
                
                # Sklearn classification report (formatted)
                logger.info(f"\n📋 Classification Report:")
                for line in class_report.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
                
                logger.info(f"="*60)
            
            # Save predictions using unified function
            result_file_path = save_predictions_csv(
                predictions, true_labels, original_split, f1, 
                dirs['result'], config.id2label, experiment_name, split_name
            )
            if logger:
                logger.debug(f"💾 {split_name.capitalize()} results saved to {result_file_path}")
            
            return f1, accuracy, confusion_matrix, result_file_path, per_class_metrics, {
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'per_class_precision': precision.tolist(),  # Convert numpy array to list
                'per_class_recall': recall.tolist(),        # Convert numpy array to list
                'per_class_f1': f1_per_class.tolist(),      # Convert numpy array to list
                'per_class_support': support.tolist(),      # Convert numpy array to list
                'classification_report': class_report
            }
        
        # Evaluate on validation and test sets
        val_f1, val_accuracy, val_confusion, val_file, val_per_class_metrics, val_detailed_metrics = evaluate_and_save_unified('val', val_dataset, dataset['val'])
        test_f1, test_accuracy, test_confusion, test_file, test_per_class_metrics, test_detailed_metrics = evaluate_and_save_unified('test', test_dataset, dataset['test'])
        
        if logger:
            logger.info(f"\n📊 FINAL RESULTS SUMMARY")
            logger.info(f"="*50)
            logger.info(f"🔍 Validation:")
            logger.info(f"  F1 Score: {val_f1:.6f}")
            logger.info(f"  Accuracy: {val_accuracy:.6f}")
            logger.info(f"  Macro Precision: {val_detailed_metrics['macro_precision']:.6f}")
            logger.info(f"  Macro Recall: {val_detailed_metrics['macro_recall']:.6f}")
            logger.info(f"🎯 Test:")
            logger.info(f"  F1 Score: {test_f1:.6f}")
            logger.info(f"  Accuracy: {test_accuracy:.6f}")
            logger.info(f"  Macro Precision: {test_detailed_metrics['macro_precision']:.6f}")
            logger.info(f"  Macro Recall: {test_detailed_metrics['macro_recall']:.6f}")
            logger.info(f"="*50)
        
        # Create results summary
        results = {
            'device': str(device),
            'n_gpu': args.n_gpu,
            'distributed': args.local_rank != -1,
            'val_f1': float(val_f1),
            'val_accuracy': float(val_accuracy),
            'test_f1': float(test_f1),
            'test_accuracy': float(test_accuracy),
            'val_confusion': val_confusion,  # Keep as numpy array for create_experiment_summary_unified
            'test_confusion': test_confusion,  # Keep as numpy array for create_experiment_summary_unified
            'experiment_hash': experiment_hash if 'experiment_hash' in locals() else 'unknown',
            'file_locations': {
                'validation_results': val_file,
                'test_results': test_file,
                'log_file': log_file if 'log_file' in locals() else '',
                'checkpoint_dir': os.path.join(dirs['output'], "checkpoints"),
            }
        }
        
        final_stats = analyze_dataset_statistics(dataset)
        weights_array = class_weights.cpu().numpy() if class_weights is not None else None
        
        # Generate training summary using unified function (native02 style)
        if not args.predict_only:
            training_summary_file = generate_training_summary_unified(
                args, results, dirs, experiment_name, 
                final_val_f1 if 'final_val_f1' in locals() else val_f1, 
                best_f1 if 'best_f1' in locals() else val_f1,
                val_confusion, test_confusion, config
            )
            if logger:
                logger.info(f"📋 Training summary saved to: {training_summary_file}")
        
        # Create experiment summary using unified function (native02 style)
        # Clean args for JSON serialization to avoid device serialization errors
        import copy
        clean_args = copy.deepcopy(args)
        
        # Remove attributes that can't be JSON serialized
        attrs_to_remove = []
        for attr_name in dir(clean_args):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(clean_args, attr_name)
                    import json
                    json.dumps(attr_value)
                except (TypeError, ValueError):
                    attrs_to_remove.append(attr_name)
        
        for attr_name in attrs_to_remove:
            if hasattr(clean_args, attr_name):
                delattr(clean_args, attr_name)
        
        summary_file, json_summary_file = create_experiment_summary_unified(
            clean_args, final_stats, results, dirs, experiment_name,
            method="sequence", class_weights=weights_array, config=config
        )
        
        # Register experiment using unified function (native02 style)
        if not args.predict_only:
            best_model_path = os.path.join(dirs['output'], 'checkpoints', 'checkpoint-epoch-best')
            if os.path.exists(best_model_path):
                register_experiment_unified(
                    experiment_name, "sequence", args.model_id, results, 
                    best_model_path, model_config, experiment_hash
                )
                
                # Register best model using unified function
                register_best_model_unified(
                    args.model_id, "sequence", args, val_f1, test_f1, 
                    experiment_name, dirs['output'], config
                )
                
                if logger:
                    logger.info("📝 Experiment and best model registered using unified system")
        
        if logger:
            logger.info("🎉 UNIFIED EXPERIMENT COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"📁 Summary saved to: {summary_file}")
            logger.info(f"📄 JSON summary saved to: {json_summary_file}")
            logger.info(f"📊 All outputs in: {dirs['output']}")
            logger.info("=" * 80)


if __name__ == "__main__":
    main() 