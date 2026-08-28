"""
Multilingual Text Classification for Alzheimer Detection via Prompt-based Fine-tuning.
Native PyTorch Implementation with ProFiT-style MLM Loss Computation.
======================================================================================
This script uses pure PyTorch training loops with ProFiT-style prompt-based learning
where MLM labels are used to locate [MASK] positions and verbalizer logits are extracted
for classification.
"""
#优化输出逻辑和checkpoint保存

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
from tqdm import tqdm

from datasets import Dataset, DatasetDict, load_dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# Import our common utilities
from common_utils import (
    ExperimentConfig, set_random_seed, create_unified_experiment_dirs,
    create_unified_external_eval_dirs, create_unified_prediction_dirs,
    analyze_dataset_statistics, print_dataset_statistics, save_dataset_statistics,
    filter_dataset_by_language, save_predictions_csv, 
    validate_verbalizer_tokens, save_model_config, load_model_config, 
    create_unified_experiment_hash,
    # New unified logging functions
    setup_unified_logging, log_epoch_progress_unified, generate_training_summary_unified,
    # New unified registration functions
    register_experiment_unified, register_best_model_unified, create_experiment_summary_unified,
    # New unified checkpoint function
    save_checkpoint_unified
)

# Set environment variable to suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptDataset(torch.utils.data.Dataset):
    """ProFiT-style dataset for prompt-based learning with proper MLM labels."""
    
    def __init__(self, examples, tokenizer, prompt_template, max_length, label2id):
        self.examples = examples
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.label2id = label2id
        
        # Pre-process the prompt template
        self.processed_prompt = prompt_template.replace("[MASK]", tokenizer.mask_token)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        label = example['label']
        
        # Tokenize text + prompt
        tokenized = self.tokenizer(
            text,
            self.processed_prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create MLM labels (ProFiT style)
        input_ids = tokenized['input_ids'].squeeze()
        mlm_labels = torch.full_like(input_ids, -1)  # Initialize with -1 (ignore)
        
        # Find [MASK] positions and mark them for loss computation
        mask_positions = (input_ids == self.tokenizer.mask_token_id)
        mlm_labels[mask_positions] = 1  # Mark [MASK] positions as 1 (participate in loss)
        
        # Convert string label to integer
        class_label = self.label2id[label]
        
        result = {
            'input_ids': input_ids,
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'mlm_labels': mlm_labels,
            'labels': torch.tensor(class_label, dtype=torch.long)
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in tokenized:
            result['token_type_ids'] = tokenized['token_type_ids'].squeeze()
        
        return result


def create_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
    """Create memory-efficient dataloader."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin memory to save GPU memory
        drop_last=False
    )


def compute_prompt_loss(model, batch, verbalizer_token_ids, class_weights=None, device='cuda'):
    """
    Compute prompt-based classification loss using ProFiT-style approach.
    
    Key changes from original:
    1. mlm_labels are used only to locate [MASK] positions (not for MLM task)
    2. Extract logits at [MASK] positions
    3. Filter to verbalizer token positions only
    4. Use classification loss on verbalizer logits
    """
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    mlm_labels = batch['mlm_labels'].to(device)
    class_labels = batch['labels'].to(device)
    
    # Add token_type_ids if available
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    if 'token_type_ids' in batch:
        model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)
    
    # Forward pass through MLM model
    outputs = model(**model_inputs)
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # ProFiT-style: Find [MASK] positions using mlm_labels
    mask_positions = (mlm_labels >= 0)  # Positions marked for loss computation
    
    # Extract logits at [MASK] positions
    masked_logits = logits[mask_positions]  # [num_masks, vocab_size]
    
    # Ensure we have the right number of masked positions
    batch_size = input_ids.size(0)
    if masked_logits.size(0) != batch_size:
        raise ValueError(f"Expected {batch_size} masked positions, got {masked_logits.size(0)}. "
                        f"Each sample should have exactly one [MASK] token.")
    
    # Extract verbalizer token logits (ProFiT-style classification)
    verbalizer_tensor = torch.tensor(verbalizer_token_ids, device=device, dtype=torch.long)
    class_logits = masked_logits[:, verbalizer_tensor]  # [batch_size, num_classes]
    
    # Compute classification loss
    if class_weights is not None:
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()
    
    loss = loss_fct(class_logits, class_labels)
    
    return loss, class_logits


# 移除自定义目录创建函数，现在使用 common_utils 中的统一函数


def train_epoch(model, dataloader, optimizer, scheduler, verbalizer_token_ids, 
                gradient_accumulation_steps, class_weights, device, scaler=None, logger=None):
    """Train for one epoch with ProFiT-style loss computation."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    # Reduced progress bar update frequency
    progress_bar = tqdm(dataloader, desc="Training", 
                       disable=False, leave=False, 
                       bar_format='{l_bar}{bar:20}{r_bar}')
    
    for step, batch in enumerate(progress_bar):
        # Compute loss using ProFiT-style approach
        if scaler is not None:  # Mixed precision
            with torch.cuda.amp.autocast():
                loss, _ = compute_prompt_loss(
                    model, batch, verbalizer_token_ids, class_weights, device
                )
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
        else:  # Regular precision
            loss, _ = compute_prompt_loss(
                model, batch, verbalizer_token_ids, class_weights, device
            )
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        total_loss += loss.item()
        
        # Gradient accumulation step
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == num_batches:
            if scaler is not None:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular gradient clipping and step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Clear cache less frequently
            if (step + 1) % (gradient_accumulation_steps * 8) == 0:
                torch.cuda.empty_cache()
        
        # Update progress bar less frequently
        if step % 10 == 0:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Clean up batch to free memory
        del batch, loss
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return total_loss / num_batches


def evaluate_epoch(model, dataloader, verbalizer_token_ids, class_weights, device, scaler=None):
    """Evaluate for one epoch with ProFiT-style loss computation."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", 
                           disable=False, leave=False,
                           bar_format='{l_bar}{bar:20}{r_bar}')
        
        for step, batch in enumerate(progress_bar):
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, class_logits = compute_prompt_loss(
                        model, batch, verbalizer_token_ids, class_weights, device
                    )
            else:
                loss, class_logits = compute_prompt_loss(
                    model, batch, verbalizer_token_ids, class_weights, device
                )
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(class_logits, dim=-1)
            labels = batch['labels']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Update progress bar less frequently
            if step % 20 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clean up
            del batch, loss, class_logits, predictions, labels
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return avg_loss, f1, all_predictions, all_labels


def setup_optimizer_scheduler(model, train_dataloader, args):
    """Setup optimizer and learning rate scheduler."""
    # Calculate total steps
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler, total_steps, warmup_steps



def predict_dataset(model, dataset, verbalizer_token_ids, device, batch_size=1):
    """Predict on a dataset using ProFiT-style approach."""
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Predicting", 
                           bar_format='{l_bar}{bar:20}{r_bar}')
        for batch in progress_bar:
            _, class_logits = compute_prompt_loss(
                model, batch, verbalizer_token_ids, None, device
            )
            
            predictions = torch.argmax(class_logits, dim=-1)
            labels = batch['labels']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            del batch, class_logits, predictions, labels
    
    torch.cuda.empty_cache()
    return np.array(all_predictions), np.array(all_labels)


# Logging functions moved to common_utils.py for unified system
# setup_unified_logging, log_epoch_progress_unified, generate_training_summary_unified


# These functions have been moved to common_utils.py as unified functions:
# - save_best_model_centralized -> register_best_model_unified
# - save_experiment_registry -> register_experiment_unified


def main():
    # Setup arguments
    parser = argparse.ArgumentParser(description='ProFiT-style Native PyTorch Prompt-based Fine-tuning')
    
    # Essential parameters
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data folder.')
    parser.add_argument('--output_dir', type=str, default='experiments/', help='Base path for outputs.')
    parser.add_argument("--result_dir", type=str, default='experiments/', help="Path for result files.")
    parser.add_argument("--log_path", type=str, default='experiments/', help="Path for log files.")
    
    parser.add_argument('--model_id', type=str, default='bert-base-multilingual-uncased', help='Model name.')
    parser.add_argument('--prompt_pattern', type=int, default=1, choices=[1, 2, 3, 4], help='Prompt pattern.')
    
    # Add checkpoint_id parameter for model loading
    parser.add_argument('--checkpoint_id', type=str, default=None, help='Path to model checkpoint for prediction.')
    parser.add_argument('--external_eval_name', type=str, default=None, help='Name for external evaluation dataset.')
    
    parser.add_argument('--languages', type=str, nargs='+', default=None, help='Languages to use.')
    parser.add_argument('--english_only', action='store_true', help='Use English only.')
    
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluation batch size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation.')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs.')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length.')
    
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--class_weights', action='store_true', help='Use class weights.')
    parser.add_argument('--predict_only', action='store_true', help='Only run prediction.')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage.')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    
    args = parser.parse_args()
    
    # Initialize config and device
    config = ExperimentConfig()
    args.method = 'prompt'
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    
    # Load model configuration if using checkpoint
    model_config = None
    if args.predict_only and args.checkpoint_id:
        try:
            model_config = load_model_config(args.checkpoint_id)
            print(f"📋 Loaded model configuration from {args.checkpoint_id}")
            
            # Override args with saved configuration for consistency
            if 'prompt_config' in model_config:
                args.prompt_pattern = model_config['prompt_config']['pattern_id']
                args.max_length = model_config['tokenizer_config']['max_length']
            if 'experiment_args' in model_config:
                saved_args = model_config['experiment_args']
                args.model_id = saved_args.get('model_id', args.model_id)
        except Exception as e:
            print(f"⚠️ Could not load model config: {e}, using current args")
    
    # Get prompt configuration from utils
    prompt_config = config.prompt_patterns[args.prompt_pattern]
    prompt_template = prompt_config['prompt']
    verbalizer_map = prompt_config['map']
    
    print(f"📋 Using utils prompt configuration:")
    print(f"   Pattern {args.prompt_pattern}: {prompt_template}")
    print(f"   Verbalizer: {verbalizer_map}")
    
    print(f"\n🔧 UNIFIED DIRECTORY STRUCTURE:")
    print(f"   Base: experiments/{args.method}/runs/{config.model_name_map.get(args.model_id, args.model_id.replace('/', '-'))}/")
    print(f"   Experiment naming: {{timestamp}}_p{{pattern}}_lr{{lr}}_tbs{{tbs}}_ebs{{ebs}}_gas{{gas}}")
    print(f"   Checkpoint strategy: checkpoint-epoch-best/ + 2 most recent checkpoints")
    print(f"   Best models: Empty directories with comprehensive parameter names")
    print(f"   Method-based structure supports: prompt, sequence, and future extensions")
    
    # Handle language filtering
    if args.english_only:
        args.languages = ['en']
    
    # Create unified directories for reproducibility
    if not args.predict_only:
        # Training: create unified directories with experiment hash
        base_dirs = {'output': args.output_dir, 'result': args.result_dir, 'log': args.log_path}
        try:
            dirs, timestamp, experiment_name, experiment_hash = create_unified_experiment_dirs(
                base_dirs, args, method="prompt", prompt_pattern=args.prompt_pattern, config=config
            )
            print(f"🚀 Experiment: {experiment_name}")
            print(f"🔒 Isolation Hash: {experiment_hash}")
            print(f"📁 Unified Directory: {dirs['output']}")
        except Exception as e:
            print(f"❌ Error creating directories: {e}")
            return
    else:
        # Prediction only: create unified directory structure for external evaluation
        experiment_hash = "prediction"
        base_dirs = {'output': args.output_dir, 'result': args.result_dir, 'log': args.log_path}
        
        if args.external_eval_name:
            # External dataset evaluation
            dirs, timestamp, experiment_name = create_unified_external_eval_dirs(
                base_dirs, args, args.external_eval_name, method="prompt", 
                prompt_pattern=args.prompt_pattern, config=config
            )
            model_short = config.model_name_map.get(args.model_id, args.model_id.replace('/', '-'))
            print(f"🔍 External Evaluation: {args.external_eval_name}")
            print(f"🎯 Model: {model_short} Pattern {args.prompt_pattern}")
        else:
            # Regular prediction
            dirs, timestamp, experiment_name = create_unified_prediction_dirs(
                base_dirs, args, method="prompt", config=config
            )
            print(f"🔍 Prediction: {experiment_name}")
        
        print(f"📁 Using checkpoint: {args.checkpoint_id}")
        print(f"📂 Output directory: {dirs['output']}")
    
    # Setup unified logging with level control
    log_file = os.path.join(dirs['log'], f"{experiment_name}.log")
    logger = setup_unified_logging(log_file, experiment_name, args.verbose)
    
    logger.info(f"🔧 PROFIT-STYLE PROMPT EXPERIMENT")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Prompt Pattern: {args.prompt_pattern}")
    logger.info(f"Template: {prompt_template}")
    logger.info(f"Verbalizers: {verbalizer_map}")
    logger.info(f"Mixed Precision: {args.fp16}")
    if args.predict_only:
        logger.info(f"Checkpoint: {args.checkpoint_id}")
    else:
        logger.info(f"Experiment Hash: {experiment_hash}")
        logger.info(f"Batch Size: {args.train_batch_size}")
        logger.info(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Effective Batch Size: {args.train_batch_size * args.gradient_accumulation_steps}")
    
    # Set deterministic environment
    set_random_seed(args.seed)
    logger.info(f"🌱 Random seed set to: {args.seed}")
    
    # Load dataset
    data_files = {
        'train': os.path.join(args.data_path, 'train.csv'),
        'val': os.path.join(args.data_path, 'val.csv'),
        'test': os.path.join(args.data_path, 'test.csv'),
    }
    
    logger.info("📊 Loading dataset...")
    dataset = load_dataset('csv', data_files=data_files)
    
    # Analyze dataset
    original_stats = analyze_dataset_statistics(dataset)
    if args.verbose:
        print_dataset_statistics(original_stats)
    save_dataset_statistics(original_stats, dirs['result'], experiment_name)
    
    # Load tokenizer and model
    logger.info(f"🔤 Loading tokenizer and model...")
    try:
        # Load tokenizer from checkpoint if available, otherwise from model_id
        tokenizer_path = args.checkpoint_id if args.predict_only and args.checkpoint_id else args.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}. Trying with add_prefix_space=True")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)
    
    # Validate verbalizer tokens using utils function
    logger.info("🔍 Validating verbalizer tokens...")
    verbalizer_token_ids = validate_verbalizer_tokens(tokenizer, verbalizer_map)
    verbalizer_ids_list = [verbalizer_token_ids[config.id2label[i]] for i in range(config.num_labels)]
    logger.info(f"Verbalizer IDs in order (NC, MCI, pAD): {verbalizer_ids_list}")
    
    # Verify single-token constraint
    for label, token_id in verbalizer_token_ids.items():
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        logger.debug(f"  {label} -> '{verbalizer_map[label]}' -> ID: {token_id} -> Token: '{token}'")
    
    # Filter dataset by language if specified
    if args.languages:
        logger.info(f"🌍 Filtering dataset for languages: {args.languages}")
        dataset = filter_dataset_by_language(dataset, args.languages)
        if dataset is None or len(dataset) == 0:
            logger.error("❌ No data found for specified languages!")
            return
        
        filtered_stats = analyze_dataset_statistics(dataset)
        if args.verbose:
            print("\nFILTERED DATASET:")
            print_dataset_statistics(filtered_stats)
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
        logger.info(f"⚖️ Class weights: {dict(zip(config.id2label.values(), class_weights.cpu().tolist()))}")
    
    # Load model
    if args.predict_only and args.checkpoint_id:
        model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_id).to(device)
        logger.info(f"📁 Loaded model from checkpoint: {args.checkpoint_id}")
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model_id, trust_remote_code=True).to(device)
        logger.info(f"🤖 Loaded model: {args.model_id}")
    
    logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create ProFiT-style datasets
    logger.info("🔄 Creating ProFiT-style datasets...")
    train_dataset = PromptDataset(
        dataset['train'], tokenizer, prompt_template, args.max_length, config.label2id
    )
    val_dataset = PromptDataset(
        dataset['val'], tokenizer, prompt_template, args.max_length, config.label2id
    )
    test_dataset = PromptDataset(
        dataset['test'], tokenizer, prompt_template, args.max_length, config.label2id
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset, args.train_batch_size, shuffle=True
    )
    val_dataloader = create_dataloader(
        val_dataset, args.eval_batch_size, shuffle=False
    )
    test_dataloader = create_dataloader(
        test_dataset, args.eval_batch_size, shuffle=False
    )
    
    logger.info(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Setup optimizer and scheduler (skip for prediction only)
    if not args.predict_only:
        optimizer, scheduler, total_steps, warmup_steps = setup_optimizer_scheduler(
            model, train_dataloader, args
        )
        
        logger.info(f"🔧 Training setup:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Steps per epoch: {len(train_dataloader) // args.gradient_accumulation_steps}")
        
        # Setup mixed precision if requested
        scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None
        if scaler:
            logger.info("🚀 Mixed precision training enabled")
    
    # Save model configuration
    model_config = {
        'method': 'prompt',
        'model_id': args.model_id,
        'prompt_config': {
            'pattern_id': args.prompt_pattern,
            'template': prompt_template,
            'verbalizer_map': verbalizer_map,
            'verbalizer_token_ids': verbalizer_token_ids,
            'verbalizer_ids_list': verbalizer_ids_list
        },
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
    
    # Training loop
    if not args.predict_only:
        logger.info("🚀 Starting ProFiT-style training...")
        
        best_f1 = 0.0
        patience_counter = 0
        checkpoint_dir = os.path.join(dirs['output'], "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_val_f1 = 0.0
        
        for epoch in range(args.num_epochs):
            logger.debug(f"🔄 Starting Epoch {epoch + 1}/{args.num_epochs}")
            
            # Training phase
            train_loss = train_epoch(
                model, train_dataloader, optimizer, scheduler, verbalizer_ids_list,
                args.gradient_accumulation_steps, class_weights, device, scaler, logger
            )
            
            # Validation phase
            val_loss, val_f1, _, _ = evaluate_epoch(
                model, val_dataloader, verbalizer_ids_list, class_weights, device, scaler
            )
            
            final_val_f1 = val_f1
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Check if this is the best model
            is_best = val_f1 > best_f1
            
            # Log epoch progress using unified function (one line)
            log_epoch_progress_unified(
                logger, epoch, args.num_epochs, train_loss, val_loss, val_f1,
                best_f1, patience_counter, args.early_stopping_patience, current_lr, is_best, style="simple"
            )
            
            # Early stopping and model saving
            if is_best:
                best_f1 = val_f1
                patience_counter = 0
                
                # Only mark as best, actual saving will be done after training completion
                logger.debug(f"💾 New best F1: {best_f1:.6f} (will save after training)")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"⏹️ Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save checkpoint using unified function
            save_checkpoint_unified(
                model, tokenizer, optimizer, scheduler, epoch, best_f1,
                checkpoint_dir, experiment_name, experiment_hash, is_best,
                max_recent_checkpoints=2, method="prompt", logger=logger
            )
        
        logger.info("✅ Training completed!")
        
        # Note: We'll use the model in memory for evaluation, then save only the final best model
    
    # Final evaluation
    logger.info("🔍 FINAL EVALUATION")
    
    def evaluate_and_save_native(split_name, dataset_split, original_split):
        """Evaluate and save results for a data split."""
        logger.debug(f"📊 Evaluating on {split_name} set...")
        
        predictions, true_labels = predict_dataset(
            model, dataset_split, verbalizer_ids_list, device, args.eval_batch_size
        )
        
        # Calculate F1 score
        f1 = f1_score(true_labels, predictions, average='macro')
        
        logger.info(f"{split_name.capitalize()} F1 score: {f1:.6f}")
        
        # Calculate confusion matrix
        confusion_matrix = np.zeros((config.num_labels, config.num_labels), dtype=int)
        for p, l in zip(predictions, true_labels):
            confusion_matrix[l][p] += 1
        
        # Save predictions
        result_file_path = save_predictions_csv(
            predictions, true_labels, original_split, f1, 
            dirs['result'], config.id2label, experiment_name, split_name
        )
        logger.debug(f"💾 {split_name.capitalize()} results saved to {result_file_path}")
        
        return f1, confusion_matrix, result_file_path
    
    # Evaluate on validation and test sets
    val_f1, val_confusion, val_file = evaluate_and_save_native('val', val_dataset, dataset['val'])
    test_f1, test_confusion, test_file = evaluate_and_save_native('test', test_dataset, dataset['test'])
    
    logger.info(f"📊 FINAL RESULTS")
    logger.info(f"  Validation F1 score: {val_f1:.6f}")
    logger.info(f"  Test F1 score: {test_f1:.6f}")
    
    # Create summary
    results = {
        'device': str(device),
        'val_f1': val_f1,
        'test_f1': test_f1,
        'val_confusion': val_confusion,
        'test_confusion': test_confusion,
        'experiment_hash': experiment_hash if 'experiment_hash' in locals() else 'unknown',
        'file_locations': {
            'validation_results': val_file,
            'test_results': test_file,
            'log_file': log_file,
            'checkpoint_dir': os.path.join(dirs['output'], "checkpoints"),
        }
    }
    
    # Note: best_model path is added in the centralized saving section above
    
    final_stats = analyze_dataset_statistics(dataset)
    weights_array = class_weights.cpu().numpy() if class_weights is not None else None
    
    prompt_info = {
        'pattern_id': args.prompt_pattern,
        'template': prompt_template,
        'verbalizer_map': verbalizer_map,
        'verbalizer_token_ids': verbalizer_token_ids
    }
    
    # Generate training summary using unified function
    if not args.predict_only:
        training_summary_file = generate_training_summary_unified(
            args, results, dirs, experiment_name, 
            final_val_f1 if 'final_val_f1' in locals() else val_f1, 
            best_f1 if 'best_f1' in locals() else val_f1,
            val_confusion, test_confusion, config
        )
        logger.info(f"📄 Training summary saved to: {training_summary_file}")
    
    # Create experiment summary (JSON format) using unified function
    summary_file, json_summary_file = create_experiment_summary_unified(
        args, final_stats, results, dirs, experiment_name, args.method, 
        weights_array, prompt_info, {}, config
    )
    
    # Save best model and register experiment using unified functions
    if not args.predict_only:
        try:
            # Save to centralized best_models directory using unified function
            centralized_path = register_best_model_unified(
                args.model_id, args.method, args, val_f1, test_f1, 
                experiment_name, args.output_dir, config
            )
            logger.info(f"💾 Best model saved to: {centralized_path}")
            
            # Update results with centralized path
            results['file_locations']['best_model'] = centralized_path
            
            # Register experiment using unified function (replaces both update_model_registry and save_experiment_registry)
            registry_file = register_experiment_unified(
                experiment_name, args.method, args.model_id, results,
                centralized_path, model_config, experiment_hash
            )
            logger.debug(f"📝 Updated unified model registry: {registry_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save centralized model: {e}")
            logger.info("📁 No best model saved")
    
    logger.info("🎉 PROFIT-STYLE EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.debug(f"📁 Summary saved to: {summary_file}")
    logger.debug(f"📄 JSON summary saved to: {json_summary_file}")
    logger.info(f"📊 All outputs in: {dirs['output']}")
    
    if not args.predict_only:
        logger.info(f"🔒 Experiment isolated with hash: {experiment_hash}")
        logger.info(f"💡 Use this hash to reproduce results or avoid conflicts")

    # Note: Experiment registration is now handled by register_experiment_unified above


if __name__ == "__main__":
    main()