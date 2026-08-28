"""
Common utilities for multilingual text classification experiments.
"""

import torch
import logging
import csv
import random
import os
import json
import datetime
import hashlib
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


# ================================
# CORE CONFIGURATION
# ================================

class ExperimentConfig:
    """Configuration class for experiment settings."""
    
    def __init__(self):
        # Label mappings
        self.label2id = {'NC': 0, 'MCI': 1, 'pAD': 2}
        self.id2label = {0: 'NC', 1: 'MCI', 2: 'pAD'}
        self.num_labels = 3
        
        # Default paths
        self.data_files = {
            'train': 'data/train.csv',
            'val': 'data/val.csv', 
            'test': 'data/test.csv'
        }
        
        # Model name mappings - Added new cased BERT model
        self.model_name_map = {
            'bert-base-multilingual-uncased': 'bert-multi',
            'bert-base-multilingual-cased': 'bert-multi-cased',
            'xlm-roberta-base': 'xlm-base',
            'xlm-roberta-large': 'xlm-large'
        }
        
        # Method name mappings
        self.method_name_map = {
            'sequence': 'seq',
            'prompt': 'prompt'
        }
        
        # Prompt patterns for prompt-based learning
        self.prompt_patterns = {
            1: {
            "prompt": "The severity of cognitive decline inferred from the passage is [MASK].",
            "map": {"NC": "low", "MCI": "some", "pAD": "high"}
            },
            
            2: { 
            "prompt": "The degree of cognitive impairment indicated by the passage is [MASK].",
            "map": {"NC": "low", "MCI": "some", "pAD": "high"}
            },
            
            3: {
            "prompt": "There are [MASK] signs of dementia, as inferred from this text.",
            "map": {"NC": "no", "MCI": "some", "pAD": "clear"}
            },
            
            4: {
            "prompt": "Based on this text, [MASK] signs of cognitive decline can be inferred.",
            "map": {"NC": "no", "MCI": "some", "pAD": "clear"}
            }
        }


# ================================
# NEW UNIFIED DIRECTORY FUNCTIONS ⭐
# ================================


# ================================
# 1. CORE FUNCTIONS - NATIVE TRAINING SCRIPT DEPENDENCIES
# ================================

def set_random_seed(seed: int) -> None:
    """
    Enhanced random seed setting for maximum reproducibility.
    
    This function sets all known sources of randomness to ensure
    deterministic behavior across different runs with the same seed.
    
    Args:
        seed: Random seed integer
    """
    
    print(f"🔧 Setting up deterministic environment with seed: {seed}")
    
    # CRITICAL: Validate and set environment variables
    current_pythonhashseed = os.environ.get('PYTHONHASHSEED')
    current_cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    
    # Validate PYTHONHASHSEED
    if not current_pythonhashseed:
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"⚠️  PYTHONHASHSEED set in Python to {seed} (should be set in shell script)")
    else:
        if current_pythonhashseed != str(seed):
            print(f"❌ WARNING: PYTHONHASHSEED mismatch! Expected: {seed}, Got: {current_pythonhashseed}")
            print(f"   This may affect reproducibility. Consider restarting Python.")
        else:
            print(f"✅ PYTHONHASHSEED correctly set to: {current_pythonhashseed}")
    
    # Validate CUBLAS_WORKSPACE_CONFIG
    if not current_cublas_config:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print(f"⚠️  CUBLAS_WORKSPACE_CONFIG set in Python to :4096:8 (should be set in shell script)")
    else:
        print(f"✅ CUBLAS_WORKSPACE_CONFIG already set to: {current_cublas_config}")
        # Validate the config value
        if current_cublas_config not in [':4096:8', ':16:8']:
            print(f"⚠️  Warning: CUBLAS_WORKSPACE_CONFIG should be :4096:8 or :16:8 for deterministic behavior")
            print(f"   Current value: {current_cublas_config}")
            print(f"   Updating to recommended value...")
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set all random number generators
    try:
        # Python standard library
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch CPU
        torch.manual_seed(seed)
        
        # PyTorch GPU (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        print(f"✅ All random generators seeded with: {seed}")
        
    except Exception as e:
        print(f"❌ Error setting random seeds: {e}")
        raise
    
    # CUDA-specific deterministic settings
    if torch.cuda.is_available():
        try:
            # CRITICAL: Enforce deterministic CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"✅ cuDNN: deterministic=True, benchmark=False")
            
            # Enable deterministic algorithms with proper CUBLAS configuration
            try:
                # First verify CUBLAS_WORKSPACE_CONFIG is properly set
                cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
                if cublas_config not in [':4096:8', ':16:8']:
                    print(f"🔧 Setting CUBLAS_WORKSPACE_CONFIG to :4096:8 for deterministic CuBLAS")
                    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                
                torch.use_deterministic_algorithms(True)
                print("✅ PyTorch deterministic algorithms: ENABLED (strict)")
                print(f"✅ CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
            except Exception as e:
                print(f"⚠️  Strict deterministic algorithms failed: {e}")
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    print("✅ PyTorch deterministic algorithms: ENABLED (warn_only)")
                    print(f"✅ CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
                except Exception as e2:
                    print(f"❌ Could not enable deterministic algorithms at all: {e2}")
                    print("ℹ️  Continuing without strict deterministic algorithms")
                    print("   This may affect reproducibility but training will proceed")
                    
        except Exception as e:
            print(f"❌ Error configuring CUDA deterministic settings: {e}")
    else:
        print("ℹ️  CUDA not available, skipping GPU-specific settings")
    
    # Final verification
    print(f"🔍 Final verification:")
    print(f"   Python seed: {seed}")
    print(f"   PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"✅ Deterministic environment setup complete!")


# These functions are moved to the "ADDITIONAL UTILITY FUNCTIONS" section



def create_unified_experiment_hash(args: Any, prompt_pattern: Optional[int] = None) -> str:
    """Generate experiment hash for reproducibility (unified version)."""
    key_params = {
        'model_id': getattr(args, 'model_id', None),
        'prompt_pattern': prompt_pattern,
        'lr': getattr(args, 'lr', None),
        'train_batch_size': getattr(args, 'train_batch_size', None),
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', None),
        'max_length': getattr(args, 'max_length', None),
        'seed': getattr(args, 'seed', None),
        'num_epochs': getattr(args, 'num_epochs', None),
        'warmup_ratio': getattr(args, 'warmup_ratio', None),
        'weight_decay': getattr(args, 'weight_decay', None)
    }
    
    # Remove None values and convert to string
    clean_params = {k: v for k, v in key_params.items() if v is not None}
    param_str = json.dumps(clean_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]



def create_unified_experiment_dirs(
    base_dirs: Dict[str, str], 
    args: Any, 
    method: str = "prompt",
    prompt_pattern: Optional[int] = None,
    config: Optional[ExperimentConfig] = None
) -> Tuple[Dict[str, str], str, str, str]:
    """
    Unified experiment directory creation function using native version logic with method parameter control.
    
    Creates directory structure:
    - base_path/{method}/runs/model_short/experiment_name/
    - base_path/{method}/best_models/model_short/
    - base_path/{method}/external_eval/
    - base_path/{method}/analysis/model_short/
    
    Args:
        base_dirs: Dictionary of base directories (output, result, log)
        args: Training arguments
        method: Experiment method ("prompt", "sequence", etc.)
        prompt_pattern: Pattern number for prompt-based methods (1-4)
        config: ExperimentConfig instance
        
    Returns:
        Tuple of (created_dirs, timestamp, experiment_name, experiment_hash)
    """
    if config is None:
        config = ExperimentConfig()
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model short name for directory hierarchy
    model_short = config.model_name_map.get(args.model_id, args.model_id.replace('/', '-'))
    
    # Generate experiment hash for isolation
    experiment_hash = create_unified_experiment_hash(args, prompt_pattern)
    
    # Create comprehensive experiment name based on method
    if method == "prompt" and prompt_pattern is not None:
        # ENHANCED: Create comprehensive experiment name for prompt method
        lr_str = f"lr{args.lr}".replace('.', '').replace('e-', 'e')  # 3e-5 -> lr3e5
        experiment_name = f"{timestamp}_p{prompt_pattern}_{lr_str}_tbs{args.train_batch_size}_ebs{getattr(args, 'eval_batch_size', args.train_batch_size)}_gas{getattr(args, 'gradient_accumulation_steps', 1)}"
    else:
        # ENHANCED: Standard naming for other methods with comprehensive parameters (matching prompt method)
        lr_str = f"lr{args.lr}".replace('.', '').replace('e-', 'e')
        experiment_name = f"{timestamp}_{method}_{lr_str}_tbs{args.train_batch_size}_ebs{getattr(args, 'eval_batch_size', args.train_batch_size)}_gas{getattr(args, 'gradient_accumulation_steps', 1)}"
    
    created_dirs = {}
    for dir_type, base_path in base_dirs.items():
        # NEW UNIFIED STRUCTURE: base_path/{method}/runs/model_short/experiment_name/
        full_path = os.path.join(base_path, method, "runs", model_short, experiment_name)
        os.makedirs(full_path, exist_ok=True)
        created_dirs[dir_type] = full_path
        
        # Create shared directories with method hierarchy
        os.makedirs(os.path.join(base_path, method, "best_models", model_short), exist_ok=True)
        os.makedirs(os.path.join(base_path, method, "external_eval"), exist_ok=True)
        os.makedirs(os.path.join(base_path, method, "analysis", model_short), exist_ok=True)
    
    return created_dirs, timestamp, experiment_name, experiment_hash



def create_unified_external_eval_dirs(
    base_dirs: Dict[str, str],
    args: Any,
    external_eval_name: str,
    method: str = "prompt",
    prompt_pattern: Optional[int] = None,
    config: Optional[ExperimentConfig] = None
) -> Tuple[Dict[str, str], str, str]:
    """
    Create unified directory structure for external evaluation.
    
    Args:
        base_dirs: Dictionary of base directories
        args: Training arguments  
        external_eval_name: Name of external evaluation dataset
        method: Experiment method
        prompt_pattern: Pattern number for prompt-based methods
        config: ExperimentConfig instance
        
    Returns:
        Tuple of (created_dirs, timestamp, experiment_name)
    """
    if config is None:
        config = ExperimentConfig()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = config.model_name_map.get(args.model_id, args.model_id.replace('/', '-'))
    
    if method == "prompt" and prompt_pattern is not None:
        experiment_name = f"external_eval_{model_short}_p{prompt_pattern}_{timestamp}"
        model_pattern_dir = f"{model_short}_p{prompt_pattern}_best"
    else:
        experiment_name = f"external_eval_{model_short}_{method}_{timestamp}"
        model_pattern_dir = f"{model_short}_{method}_best"
    
    created_dirs = {}
    for dir_type, base_path in base_dirs.items():
        # Structure: base_path/{method}/external_eval/{external_eval_name}/{model_pattern_dir}/
        full_path = os.path.join(base_path, method, "external_eval", external_eval_name, model_pattern_dir)
        os.makedirs(full_path, exist_ok=True)
        created_dirs[dir_type] = full_path
    
    return created_dirs, timestamp, experiment_name



def create_unified_prediction_dirs(
    base_dirs: Dict[str, str],
    args: Any,
    method: str = "prompt",
    config: Optional[ExperimentConfig] = None
) -> Tuple[Dict[str, str], str, str]:
    """
    Create unified directory structure for general prediction.
    
    Args:
        base_dirs: Dictionary of base directories
        args: Training arguments
        method: Experiment method
        config: ExperimentConfig instance
        
    Returns:
        Tuple of (created_dirs, timestamp, experiment_name)
    """
    if config is None:
        config = ExperimentConfig()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"predict_{args.model_id.replace('/', '_')}_{method}_{timestamp}"
    
    created_dirs = {}
    for dir_type, base_path in base_dirs.items():
        # Structure: base_path/predictions/{method}/
        full_path = os.path.join(base_path, "predictions", method)
        os.makedirs(full_path, exist_ok=True)
        created_dirs[dir_type] = full_path
    
    return created_dirs, timestamp, experiment_name



def setup_unified_logging(
    log_file: str, 
    experiment_name: str, 
    verbose: bool = False,
    filter_http: bool = True
) -> logging.Logger:
    """
    Unified logging setup function integrating best features from both versions.
    
    Args:
        log_file: Path to log file
        experiment_name: Name of the experiment for logger identification
        verbose: Enable DEBUG level logging (default: INFO only)
        filter_http: Filter out HTTP request logs (default: True)
        
    Returns:
        Configured logger instance
    """
    
    class InfoFilter(logging.Filter):
        def filter(self, record):
            if not filter_http:
                return True
            return 'HTTP Request' not in record.getMessage()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(detailed_formatter)
    if filter_http:
        file_handler.addFilter(InfoFilter())
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    if filter_http:
        console_handler.addFilter(InfoFilter())
    console_handler.setLevel(logging.INFO)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



def validate_verbalizer_tokens(tokenizer, verbalizer_map: Dict[str, str]) -> Dict[str, int]:
    """Validate and convert verbalizer words to token IDs."""
    verbalizer_ids = {}
    for label, word in verbalizer_map.items():
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"Verbalizer word '{word}' for label '{label}' is UNK token. Please choose another word.")
        verbalizer_ids[label] = token_id
    return verbalizer_ids



def save_model_config(model_config: Dict[str, Any], save_path: str) -> str:
    """Save complete model configuration for later loading."""
    config_file = os.path.join(save_path, 'model_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    return config_file



def load_model_config(model_path: str) -> Dict[str, Any]:
    """Load model configuration from saved file."""
    config_file = os.path.join(model_path, 'model_config.json')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Model config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)




# ================================
# 2. EXPERIMENT MANAGEMENT FUNCTIONS
# ================================

def register_experiment_unified(
    experiment_name: str,
    method: str, 
    model_id: str,
    results: Dict[str, Any],
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_hash: Optional[str] = None,
    base_registry_dir: str = 'experiments/'
) -> str:
    """
    Unified experiment registration function integrating all tracking functionality.
    
    This function replaces both update_model_registry() and save_experiment_registry()
    to provide a single, comprehensive registration system.
    
    Args:
        experiment_name: Name of the experiment
        method: Experiment method ("prompt", "sequence", etc.)
        model_id: Model identifier
        results: Dictionary containing experiment results
        model_path: Path to the saved model (optional)
        config: Model/experiment configuration (optional)
        experiment_hash: Experiment hash for reproducibility (optional)
        base_registry_dir: Base directory for registry files
        
    Returns:
        Path to the updated registry file
    """
    
    # Create method-specific registry file
    registry_file = os.path.join(base_registry_dir, f"{method}_model_registry.json")
    
    # Create registry directory if it doesn't exist
    registry_dir = os.path.dirname(registry_file)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)
    
    # Load existing registry or create new one
    if os.path.exists(registry_file):
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        registry = {
            'method': method,
            'experiments': {},  # Dictionary by experiment_name for compatibility
            'experiments_list': [],  # List for chronological tracking
            'best_models': {
                'by_test_f1': None,
                'by_val_f1': None,
                'by_model': {}  # Best results for each model
            },
            'statistics': {
                'total_experiments': 0,
                'models_tested': [],
                'best_test_f1': 0.0,
                'best_val_f1': 0.0
            }
        }
    
    # Create comprehensive experiment entry with JSON-serializable results
    serializable_results = {}
    for key, value in results.items():
        if hasattr(value, 'tolist'):  # Handle numpy arrays
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    experiment_entry = {
        'experiment_name': experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'method': method,
        'model_id': model_id,
        'model_path': model_path,
        'test_f1': results.get('test_f1', 0.0),
        'val_f1': results.get('val_f1', 0.0),
        'experiment_hash': experiment_hash,
        'config': config if config else {},
        'results': serializable_results,
        'prompt_pattern': config.get('prompt_config', {}).get('pattern_id') if config and method == 'prompt' else None
    }
    
    # Add to both dictionary (for backward compatibility) and list (for ordering)
    registry['experiments'][experiment_name] = experiment_entry
    registry['experiments_list'].append(experiment_entry)
    
    # Update best models by test F1
    current_best_test = registry['best_models']['by_test_f1']
    if not current_best_test or experiment_entry['test_f1'] > current_best_test.get('test_f1', 0.0):
        registry['best_models']['by_test_f1'] = experiment_entry
        registry['statistics']['best_test_f1'] = experiment_entry['test_f1']
    
    # Update best models by validation F1
    current_best_val = registry['best_models']['by_val_f1']
    if not current_best_val or experiment_entry['val_f1'] > current_best_val.get('val_f1', 0.0):
        registry['best_models']['by_val_f1'] = experiment_entry
        registry['statistics']['best_val_f1'] = experiment_entry['val_f1']
    
    # Update best model for this specific model_id
    model_key = model_id.replace('/', '_')
    current_model_best = registry['best_models']['by_model'].get(model_key)
    if not current_model_best or experiment_entry['test_f1'] > current_model_best.get('test_f1', 0.0):
        registry['best_models']['by_model'][model_key] = experiment_entry
    
    # Update statistics
    registry['statistics']['total_experiments'] = len(registry['experiments_list'])
    models_tested = list(set([exp['model_id'] for exp in registry['experiments_list']]))
    registry['statistics']['models_tested'] = models_tested
    
    # Save updated registry
    with open(registry_file, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    return registry_file



def register_best_model_unified(
    model_id: str,
    method: str,
    args: Any,
    val_f1: float,
    test_f1: float,
    experiment_name: Optional[str] = None,
    base_output_dir: str = "experiments",
    config: Optional[ExperimentConfig] = None
) -> str:
    """
    Unified best model registration function integrating save_best_model_centralized logic.
    
    Creates enhanced best model directory structure with comprehensive metadata.
    
    Args:
        model_id: Model identifier
        method: Experiment method
        args: Training arguments
        val_f1: Validation F1 score
        test_f1: Test F1 score
        experiment_name: Name of the experiment (optional)
        base_output_dir: Base output directory
        config: ExperimentConfig instance
        
    Returns:
        Path to the created best model directory
    """
    if config is None:
        config = ExperimentConfig()
    
    model_short = config.model_name_map.get(model_id, model_id.replace('/', '-'))
    
    # Create best_models directory structure with unified method structure
    best_models_base = os.path.join(base_output_dir, method, "best_models", model_short)
    os.makedirs(best_models_base, exist_ok=True)
    
    # ENHANCED: Save pattern-specific best model with comprehensive parameters
    if method == "prompt" and hasattr(args, 'prompt_pattern'):
        pattern_dir = f"p{args.prompt_pattern}_best_val{val_f1:.4f}_test{test_f1:.4f}_tbs{args.train_batch_size}_ebs{getattr(args, 'eval_batch_size', args.train_batch_size)}_gas{getattr(args, 'gradient_accumulation_steps', 1)}"
        prompt_pattern = args.prompt_pattern
    else:
        # ENHANCED: Include gas parameter for sequence and other methods (matching prompt method)
        pattern_dir = f"{method}_best_val{val_f1:.4f}_test{test_f1:.4f}_tbs{args.train_batch_size}_ebs{getattr(args, 'eval_batch_size', args.train_batch_size)}_gas{getattr(args, 'gradient_accumulation_steps', 1)}"
        prompt_pattern = None
    
    pattern_path = os.path.join(best_models_base, pattern_dir)
    
    # Create empty directory structure (no model files to save space)
    if not os.path.exists(pattern_path):
        os.makedirs(pattern_path, exist_ok=True)
        
        # OPTIMIZATION: Only save metadata, not model files (save storage space)
        metadata = {
            "model_id": model_id,
            "model_short": model_short,
            "method": method,
            "prompt_pattern": prompt_pattern,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": getattr(args, 'eval_batch_size', args.train_batch_size),
            "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
            "learning_rate": getattr(args, 'lr', None),
            "max_length": getattr(args, 'max_length', None),
            "experiment_name": experiment_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "note": "Model files are stored in the experiment's checkpoint directory to save space",
            "checkpoint_reference": "Check experiment runs directory for actual model files"
        }
        
        # Save metadata file
        with open(os.path.join(pattern_path, "model_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Created empty best model directory: {pattern_dir}")
        print(f"💡 Model files remain in checkpoint directory to save storage space")
        
        # Update model registry with unified method structure
        registry_file = os.path.join(base_output_dir, method, "best_models", "model_registry.json")
        if os.path.exists(registry_file):
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        if model_short not in registry:
            registry[model_short] = {"patterns": {}, "overall_best": None}
        
        # Update pattern-specific best with enhanced information
        pattern_key = str(prompt_pattern) if prompt_pattern is not None else method
        registry[model_short]["patterns"][pattern_key] = {
            "path": pattern_dir,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "experiment_name": experiment_name or f"{model_short}_{pattern_key}",
            "timestamp": datetime.datetime.now().isoformat(),
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": getattr(args, 'eval_batch_size', args.train_batch_size),
            "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
            "learning_rate": getattr(args, 'lr', None),
            "max_length": getattr(args, 'max_length', None),
            "method": method
        }
        
        # Update overall best for this model
        current_overall = registry[model_short]["overall_best"]
        if not current_overall or test_f1 > current_overall.get("test_f1", 0):
            registry[model_short]["overall_best"] = {
                "path": pattern_dir,
                "pattern": prompt_pattern if prompt_pattern is not None else method,
                "val_f1": val_f1,
                "test_f1": test_f1,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": getattr(args, 'eval_batch_size', args.train_batch_size),
                "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
                "learning_rate": getattr(args, 'lr', None),
                "method": method
            }
        
        # Save updated registry
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        
        return pattern_path
    
    return pattern_path



def create_experiment_summary_unified(
    args: Any, 
    stats: Dict[str, Any], 
    results: Dict[str, Any], 
    dirs: Dict[str, str], 
    experiment_name: str,
    method: str = "prompt",
    class_weights: Optional[np.ndarray] = None,
    prompt_info: Optional[Dict[str, Any]] = None,
    length_stats: Optional[Dict[str, Any]] = None,
    config: Optional[ExperimentConfig] = None
) -> Tuple[str, str]:
    """
    Unified experiment summary creation function with enhanced structure and method support.
    
    Args:
        args: Training arguments
        stats: Dataset statistics
        results: Training results
        dirs: Directory paths
        experiment_name: Name of the experiment
        method: Experiment method
        class_weights: Class weights array (optional)
        prompt_info: Prompt information (optional)
        length_stats: Text length statistics (optional)
        config: ExperimentConfig instance
        
    Returns:
        Tuple of (text_summary_file, json_summary_file)
    """
    if config is None:
        config = ExperimentConfig()
    
    # Text summary
    summary_file = os.path.join(dirs['result'], f"{experiment_name}_experiment_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("UNIFIED EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Experiment Information:\n")
        f.write(f"  Experiment Name: {experiment_name}\n")
        f.write(f"  Model ID: {args.model_id}\n")
        f.write(f"  Method: {method}\n")
        
        if method == "prompt" and prompt_info:
            f.write(f"  Prompt Pattern: {prompt_info['pattern_id']}\n")
            f.write(f"  Prompt Template: {prompt_info['template']}\n")
            f.write(f"  Verbalizer Map: {prompt_info['verbalizer_map']}\n")
        
        f.write(f"  Languages: {getattr(args, 'languages', None) if getattr(args, 'languages', None) else 'All languages'}\n")
        f.write(f"  Device: {results.get('device', 'unknown')}\n")
        f.write(f"  Experiment Hash: {results.get('experiment_hash', 'unknown')}\n\n")

        f.write("Reproducibility Information:\n")
        f.write(f"  Random Seed: {getattr(args, 'seed', 'unknown')}\n")
        f.write(f"  PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}\n")
        f.write(f"  CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}\n")
        f.write(f"  PyTorch Deterministic: {'Enabled' if torch.are_deterministic_algorithms_enabled() else 'Disabled'}\n")
        f.write(f"  cuDNN Deterministic: {'Enabled' if torch.backends.cudnn.deterministic else 'Disabled'}\n")
        f.write(f"  cuDNN Benchmark: {'Disabled' if not torch.backends.cudnn.benchmark else 'Enabled'}\n\n")

        f.write("Training Parameters:\n")
        for key, value in vars(args).items():
            if key not in ['data_path', 'output_dir', 'result_dir', 'log_path']:
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Dataset Information:\n")
        for split in ['train', 'val', 'test']:
            if split in stats:
                count = stats[split]['total_samples']
                dist = stats[split]['label_distribution']
                f.write(f"  {split.capitalize()} samples: {count}\n")
                f.write(f"  {split.capitalize()} distribution: {dist}\n")
        f.write("\n")
        
        # Add text length statistics
        if length_stats:
            f.write("Text Length Statistics:\n")
            for split_name, split_stats in length_stats.items():
                f.write(f"  {split_name.upper()} SET:\n")
                f.write(f"    Total samples: {split_stats['total_samples']}\n")
                f.write(f"    Mean length: {split_stats['mean_length']:.1f} tokens\n")
                f.write(f"    Median length: {split_stats['median_length']:.1f} tokens\n")
                f.write(f"    95th percentile: {split_stats['percentile_95']:.1f} tokens\n")
                f.write(f"    Max length: {split_stats['max_length']} tokens\n")
                f.write(f"    Min length: {split_stats['min_length']} tokens\n")
                f.write(f"    Texts > 256 tokens: {split_stats['over_256_ratio']:.1%}\n")
                f.write(f"    Texts > 512 tokens: {split_stats['over_512_ratio']:.1%}\n")
                f.write(f"    Texts > 768 tokens: {split_stats['over_768_ratio']:.1%}\n")
            f.write("\n")
        
        f.write("Results:\n")
        f.write(f"  Validation F1 Score: {results['val_f1']:.4f}\n")
        f.write(f"  Test F1 Score: {results['test_f1']:.4f}\n\n")

        # Confusion matrices
        for split in ['val', 'test']:
            confusion = results[f'{split}_confusion']
            f.write(f"{split.capitalize()} Confusion Matrix (rows=true, cols=pred):\n")
            f.write("Labels: " + " ".join([f"{config.id2label[i]:>8}" for i in range(len(config.id2label))]) + "\n")
            for i, row in enumerate(confusion):
                f.write(f"{config.id2label[i]:>6}: " + " ".join([f"{val:>8}" for val in row]) + "\n")
            f.write("\n")
            
            # Per-class metrics
            metrics = compute_per_class_metrics(confusion, config.id2label)
            f.write(f"{split.capitalize()} Per-class Metrics:\n")
            for label, metric in metrics.items():
                f.write(f"  {label:>6}: Precision={metric['precision']:.4f}, "
                       f"Recall={metric['recall']:.4f}, F1={metric['f1']:.4f}\n")
            f.write("\n")
        
        if class_weights is not None:
            f.write("Class Weights:\n")
            for i, weight in enumerate(class_weights):
                f.write(f"  {config.id2label[i]:>6}: {weight:.4f}\n")
            f.write("\n")

        f.write("Output Files:\n")
        for key, path in results.get('file_locations', {}).items():
            f.write(f"  {key}: {path}\n")

    # Enhanced JSON summary with unified structure
    json_summary_file = os.path.join(dirs['result'], f"{experiment_name}_experiment_summary.json")
    summary_data = {
        'experiment_info': {
            'experiment_name': experiment_name,
            'model_id': args.model_id,
            'method': method,
            'languages': getattr(args, 'languages', None) if getattr(args, 'languages', None) else 'all',
            'device': results.get('device', 'unknown'),
            'experiment_hash': results.get('experiment_hash', 'unknown'),
            'parameters': vars(args)
        },
        'dataset_statistics': stats,
        'results': {
            'validation_f1': results['val_f1'],
            'test_f1': results['test_f1'],
            'validation_confusion_matrix': results['val_confusion'].tolist() if hasattr(results['val_confusion'], 'tolist') else results['val_confusion'],
            'test_confusion_matrix': results['test_confusion'].tolist() if hasattr(results['test_confusion'], 'tolist') else results['test_confusion']
        },
        'file_locations': results.get('file_locations', {}),
        'reproducibility_info': {
            'experiment_hash': results.get('experiment_hash', 'unknown'),
            'random_seed': getattr(args, 'seed', None),
            'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'Not set'),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set'),
            'pytorch_deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark
        },
        'unified_structure': {
            'method': method,
            'directory_system': 'unified',
            'registration_system': 'unified',
            'checkpoint_system': 'unified',
            'summary_system': 'unified',
            'created_timestamp': datetime.datetime.now().isoformat(),
            'version': '2.0'
        }
    }
    
    if prompt_info:
        summary_data['prompt_info'] = prompt_info
        
    if length_stats:
        summary_data['text_length_statistics'] = length_stats
        
    if class_weights is not None:
        summary_data['class_weights'] = class_weights.tolist()
    
    with open(json_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return summary_file, json_summary_file



def save_checkpoint_unified(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    best_f1: float,
    checkpoint_dir: str,
    experiment_name: str,
    experiment_hash: str,
    is_best: bool = False,
    max_recent_checkpoints: int = 2,
    method: str = "prompt",
    logger: Optional[Any] = None
) -> str:
    """
    Unified checkpoint saving function with enhanced configuration options.
    
    This function provides a unified checkpoint management strategy:
    - Best checkpoint: checkpoint-epoch-best/ (replaces previous best)
    - Recent checkpoints: Keep configurable number of most recent checkpoints
    - Automatic cleanup of old checkpoints to save storage space
    
    Args:
        model: PyTorch model to save
        tokenizer: Tokenizer to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        epoch: Current epoch number
        best_f1: Best F1 score achieved so far
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        experiment_hash: Experiment hash for reproducibility
        is_best: Whether this is the best checkpoint so far
        max_recent_checkpoints: Maximum number of recent checkpoints to keep (default: 2)
        method: Training method ("prompt", "sequence", etc.)
        logger: Optional logger for enhanced logging
        
    Returns:
        Path to the saved checkpoint directory
    """
    import glob
    import shutil
    import torch
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_func = logger.info if logger else print
    
    if is_best:
        # Save best checkpoint with special name
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-epoch-best")
        
        # Remove old best checkpoint if exists
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
            log_func("🔄 Replaced previous best checkpoint")
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save comprehensive training state
        training_state = {
            'epoch': epoch,
            'best_f1': best_f1,
            'experiment_name': experiment_name,
            'experiment_hash': experiment_hash,
            'method': method,
            'is_best': True,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'save_timestamp': datetime.datetime.now().isoformat()
        }
        
        torch.save(training_state, os.path.join(checkpoint_path, 'training_state.pt'))
        
        log_func(f"💾 Saved BEST checkpoint: checkpoint-epoch-best (F1: {best_f1:.4f})")
        saved_checkpoint_path = checkpoint_path
    else:
        saved_checkpoint_path = None
    
    # Always save regular checkpoint (for recent history)
    regular_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}")
    
    # Avoid duplicate saving if this epoch is also the best
    if not os.path.exists(regular_checkpoint_path):
        os.makedirs(regular_checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(regular_checkpoint_path)
        tokenizer.save_pretrained(regular_checkpoint_path)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'best_f1': best_f1,
            'experiment_name': experiment_name,
            'experiment_hash': experiment_hash,
            'method': method,
            'is_best': False,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'save_timestamp': datetime.datetime.now().isoformat()
        }
        
        torch.save(training_state, os.path.join(regular_checkpoint_path, 'training_state.pt'))
        
        if not is_best:  # Only set return path if this isn't already the best
            saved_checkpoint_path = regular_checkpoint_path
    
    # Clean up old regular checkpoints based on configuration
    if max_recent_checkpoints > 0:
        checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-epoch-[0-9]*")
        existing_checkpoints = [cp for cp in glob.glob(checkpoint_pattern) 
                               if not cp.endswith("checkpoint-epoch-best")]
        
        if len(existing_checkpoints) > max_recent_checkpoints:
            # Sort by epoch number (extract from folder name)
            def extract_epoch(path):
                try:
                    return int(os.path.basename(path).split('-')[-1])
                except (ValueError, IndexError):
                    return 0
            
            existing_checkpoints.sort(key=extract_epoch)
            
            # Remove oldest checkpoints, keep only the most recent ones
            checkpoints_to_remove = existing_checkpoints[:-max_recent_checkpoints]
            
            for old_checkpoint in checkpoints_to_remove:
                if os.path.exists(old_checkpoint):
                    shutil.rmtree(old_checkpoint)
                    epoch_num = extract_epoch(old_checkpoint)
                    log_func(f"🗑️ Removed old checkpoint: checkpoint-epoch-{epoch_num}")
    
    # Log checkpoint summary
    remaining_checkpoints = [cp for cp in glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*"))]
    best_exists = os.path.exists(os.path.join(checkpoint_dir, "checkpoint-epoch-best"))
    
    total_checkpoints = len(remaining_checkpoints) + (1 if best_exists else 0)
    log_func(f"📊 Checkpoint summary: {total_checkpoints} total "
             f"(1 best + {len(remaining_checkpoints)} recent)")
    
    return saved_checkpoint_path if saved_checkpoint_path else regular_checkpoint_path




# ================================
# 3. DATASET PROCESSING FUNCTIONS
# ================================

def analyze_dataset_statistics(dataset: DatasetDict, 
                             split_names: List[str] = ['train', 'val', 'test']) -> Dict[str, Any]:
    """Analyze dataset statistics by label and language."""
    stats = {}
    for split in split_names:
        if split not in dataset:
            continue
            
        split_data = dataset[split]
        total_samples = len(split_data)
        label_counts = Counter(item['label'] for item in split_data)
        lang_stats = defaultdict(lambda: defaultdict(int))
        lang_totals = defaultdict(int)
        
        for item in split_data:
            lang = item.get('lang', 'unknown')
            label = item['label']
            lang_stats[lang][label] += 1
            lang_totals[lang] += 1
        
        stats[split] = {
            'total_samples': total_samples,
            'label_distribution': dict(label_counts),
            'language_distribution': dict(lang_totals),
            'lang_label_matrix': {lang: dict(labels) for lang, labels in lang_stats.items()}
        }
    return stats



def print_dataset_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted dataset statistics."""
    print("\n" + "="*80 + "\nDATASET STATISTICS\n" + "="*80)
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} SET (Total: {split_stats['total_samples']})")
        
        # Label distribution
        label_dist = ", ".join([
            f"{l}: {c} ({c/split_stats['total_samples']:.1%})" 
            for l, c in split_stats['label_distribution'].items()
        ])
        print(f"Label distribution: {label_dist}")
        
        # Language distribution 
        lang_dist = ", ".join([
            f"{l}: {c} ({c/split_stats['total_samples']:.1%})" 
            for l, c in split_stats['language_distribution'].items()
        ])
        print(f"Language distribution: {lang_dist}")
        
        # Matrix
        print(f"\n{'Lang':<6} {'NC':<6} {'MCI':<6} {'pAD':<6} {'Total':<6}")
        print("-" * 36)
        for lang, label_counts in split_stats['lang_label_matrix'].items():
            nc, mci, pad = label_counts.get('NC', 0), label_counts.get('MCI', 0), label_counts.get('pAD', 0)
            print(f"{lang:<6} {nc:<6} {mci:<6} {pad:<6} {nc+mci+pad:<6}")



def save_dataset_statistics(stats: Dict[str, Any], save_path: str, experiment_name: str) -> None:
    """Save dataset statistics to JSON file."""
    json_path = os.path.join(save_path, f'{experiment_name}_dataset_statistics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)



def filter_dataset_by_language(dataset: DatasetDict, 
                              target_languages: Optional[Union[str, List[str]]] = None) -> Optional[DatasetDict]:
    """Filter dataset by specified languages."""
    if not target_languages:
        return dataset
        
    if isinstance(target_languages, str):
        target_languages = [target_languages]
    
    filtered_dataset = DatasetDict()
    for split_name, split_data in dataset.items():
        filtered_data = split_data.filter(lambda x: x.get('lang', 'unknown') in target_languages)
        if len(filtered_data) > 0:
            filtered_dataset[split_name] = filtered_data
            
    return filtered_dataset if filtered_dataset else None



def compute_per_class_metrics(confusion_matrix: np.ndarray, id2label: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1 scores."""
    metrics = {}
    num_labels = len(id2label)
    
    for i in range(num_labels):
        true_positives = confusion_matrix[i, i]
        actual_positives = confusion_matrix[i, :].sum()
        predicted_positives = confusion_matrix[:, i].sum()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[id2label[i]] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return metrics


# ================================
# LOGGING AND EXPERIMENT SUMMARY (NEW UNIFIED FUNCTIONS)
# ================================



# ================================
# 4. LOGGING AND PROGRESS FUNCTIONS
# ================================

def log_epoch_progress_unified(
    logger: logging.Logger, 
    epoch: int, 
    total_epochs: int,
    train_loss: float, 
    val_loss: Optional[float], 
    val_f1: float,
    best_f1: float, 
    patience_counter: int, 
    patience_limit: int,
    learning_rate: float, 
    is_best: bool = False,
    style: str = "simple"
) -> None:
    """
    Unified epoch progress logging function supporting both detailed and simple styles.
    
    Args:
        logger: Logger instance
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss (optional for simple style)
        val_f1: Validation F1 score
        best_f1: Best F1 score so far
        patience_counter: Current patience counter
        patience_limit: Early stopping patience limit
        learning_rate: Current learning rate
        is_best: Whether this is the best epoch
        style: "simple" for one-line or "detailed" for multi-line output
    """
    
    if style == "detailed":
        # Multi-line detailed style
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch + 1}/{total_epochs} COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Training Loss: {train_loss:.6f}")
        logger.info(f"Validation F1: {val_f1:.4f}")
        logger.info(f"Best F1 so far: {best_f1:.4f}")
        logger.info(f"Learning Rate: {learning_rate:.2e}")
        
        if is_best:
            logger.info("🎉 NEW BEST MODEL! Saving checkpoint...")
        else:
            logger.info(f"⏳ Early stopping patience: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                logger.info("🛑 EARLY STOPPING TRIGGERED!")
        logger.info("=" * 80)
        
    else:  # simple style (default)
        # Single-line compact style
        status = "🎉 NEW BEST!" if is_best else f"Patience: {patience_counter}/{patience_limit}"
        
        if val_loss is not None:
            # Include validation loss if provided
            logger.info(f"Epoch {epoch+1:2d}/{total_epochs} | "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                       f"F1: {val_f1:.4f} | Best: {best_f1:.4f} | "
                       f"LR: {learning_rate:.2e} | {status}")
        else:
            # Fallback without validation loss
            logger.info(f"Epoch {epoch+1:2d}/{total_epochs} | "
                       f"Train: {train_loss:.4f} | "
                       f"F1: {val_f1:.4f} | Best: {best_f1:.4f} | "
                       f"LR: {learning_rate:.2e} | {status}")



def generate_training_summary_unified(
    args: Any, 
    results: Dict[str, Any], 
    dirs: Dict[str, str], 
    experiment_name: str,
    final_val_f1: float, 
    best_f1: float, 
    val_confusion: np.ndarray, 
    test_confusion: np.ndarray,
    config: ExperimentConfig
) -> str:
    """
    Generate comprehensive training summary file (moved from main training script).
    
    Args:
        args: Training arguments
        results: Training results dictionary
        dirs: Directory paths
        experiment_name: Name of the experiment
        final_val_f1: Final validation F1 score
        best_f1: Best validation F1 score achieved
        val_confusion: Validation confusion matrix
        test_confusion: Test confusion matrix
        config: ExperimentConfig instance
        
    Returns:
        Path to the generated summary file
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_content = f"""TRAINING SUMMARY - {experiment_name}
{'='*80}
Generated: {timestamp}

EXPERIMENT CONFIGURATION:
{'='*40}
Model ID: {args.model_id}
Method: {getattr(args, 'method', 'unknown')}
Prompt Pattern: {getattr(args, 'prompt_pattern', 'N/A')}
Max Length: {args.max_length}
Learning Rate: {args.lr}
Batch Size: {args.train_batch_size}
Gradient Accumulation: {getattr(args, 'gradient_accumulation_steps', 1)}
Effective Batch Size: {args.train_batch_size * getattr(args, 'gradient_accumulation_steps', 1)}
Number of Epochs: {args.num_epochs}
Weight Decay: {getattr(args, 'weight_decay', 0.0)}
Warmup Ratio: {getattr(args, 'warmup_ratio', 0.0)}
Early Stopping Patience: {getattr(args, 'early_stopping_patience', 0)}
Mixed Precision: {getattr(args, 'fp16', False)}
Class Weights: {getattr(args, 'class_weights', False)}
Random Seed: {args.seed}

PROMPT CONFIGURATION:
{'='*40}
Template: {config.prompt_patterns[getattr(args, 'prompt_pattern', 1)]['prompt'] if hasattr(config, 'prompt_patterns') and getattr(args, 'prompt_pattern', None) else 'N/A'}
Verbalizer Map: {config.prompt_patterns[getattr(args, 'prompt_pattern', 1)]['map'] if hasattr(config, 'prompt_patterns') and getattr(args, 'prompt_pattern', None) else 'N/A'}

REPRODUCIBILITY INFORMATION:
{'='*40}
Experiment Hash: {results.get('experiment_hash', 'unknown')}
Random Seed: {args.seed}
PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}
CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}
PyTorch Deterministic: {'Enabled' if torch.are_deterministic_algorithms_enabled() else 'Disabled'}
cuDNN Deterministic: {'Enabled' if torch.backends.cudnn.deterministic else 'Disabled'}
cuDNN Benchmark: {'Disabled' if not torch.backends.cudnn.benchmark else 'Enabled'}

TRAINING RESULTS:
{'='*40}
Final Validation F1: {final_val_f1:.6f}
Best Validation F1: {best_f1:.6f}
Test F1: {results.get('test_f1', 0.0):.6f}
Training Status: {'COMPLETED (BEST MODEL)' if final_val_f1 == best_f1 else 'EARLY STOPPED'}

VALIDATION CONFUSION MATRIX:
{'='*40}
Labels: {list(config.id2label.values())}
{val_confusion}

TEST CONFUSION MATRIX:
{'='*40}
Labels: {list(config.id2label.values())}
{test_confusion}

FILE LOCATIONS:
{'='*40}
Log File: {results['file_locations']['log_file']}
Validation Results: {results['file_locations']['validation_results']}
Test Results: {results['file_locations']['test_results']}
Checkpoint Directory: {results['file_locations']['checkpoint_dir']}
{'Best Model: ' + results['file_locations']['best_model'] if 'best_model' in results['file_locations'] else 'Best Model: Not saved (not best performance)'}

{'='*80}
"""
    
    summary_file = os.path.join(dirs['output'], f"{experiment_name}_training_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    return summary_file



def save_predictions_csv(predictions: np.ndarray, labels: np.ndarray, dataset_split: Dataset,
                        f1_score: float, save_path: str, id2label: Dict[int, str], 
                        experiment_name: str, split_name: str) -> str:
    """Save predictions to CSV file and return the file path."""
    file_path = os.path.join(save_path, f"{experiment_name}_{split_name}.csv")
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'file_idx', 'prediction', 'true_label', 'text', f'f1: {f1_score:.4f}'])
        for idx, (item, pred_idx, true_idx) in enumerate(zip(dataset_split, predictions, labels)):
            writer.writerow([
                idx, 
                item.get('idx', idx), 
                id2label[pred_idx], 
                id2label[true_idx],
                item['text']
            ])
    return file_path




# ================================
# 5. REGISTRY AND MODEL MANAGEMENT FUNCTIONS
# ================================

def get_best_model_path(method: str, criterion: str = 'test_f1', model_id: Optional[str] = None,
                       base_registry_dir: str = 'experiments/') -> Optional[str]:
    """
    Get the path to the best model for a given method.
    
    Args:
        method: 'prompt' or 'sequence'
        criterion: 'test_f1', 'val_f1', or 'by_model'
        model_id: Specific model ID (only used when criterion='by_model')
        base_registry_dir: Base directory for registry files
    
    Returns:
        Path to the best model or None if not found
    """
    registry_file = os.path.join(base_registry_dir, f"{method}_model_registry.json")
    
    if not os.path.exists(registry_file):
        return None
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    if criterion == 'by_model' and model_id:
        model_key = model_id.replace('/', '_')
        best_model = registry.get('best_models', {}).get('by_model', {}).get(model_key)
    else:
        best_model = registry.get('best_models', {}).get(f'by_{criterion}')
    
    return best_model['model_path'] if best_model else None



def get_method_summary(method: str, base_registry_dir: str = 'experiments/') -> Optional[Dict[str, Any]]:
    """Get comprehensive summary for a specific method."""
    registry_file = os.path.join(base_registry_dir, f"{method}_model_registry.json")
    
    if not os.path.exists(registry_file):
        return None
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # Create summary
    summary = {
        'method': method,
        'total_experiments': len(registry.get('experiments', [])),
        'models_tested': registry.get('statistics', {}).get('models_tested', []),
        'best_test_f1': registry.get('statistics', {}).get('best_test_f1', 0.0),
        'best_val_f1': registry.get('statistics', {}).get('best_val_f1', 0.0),
        'best_models': registry.get('best_models', {}),
        'recent_experiments': sorted(
            registry.get('experiments_list', []), 
            key=lambda x: x.get('timestamp', '') if isinstance(x, dict) else '', 
            reverse=True
        )[:5]  # Last 5 experiments
    }
    
    return summary



def compare_methods(base_registry_dir: str = 'experiments/') -> Dict[str, Any]:
    """Compare performance across different methods."""
    methods = ['prompt', 'sequence']
    comparison = {
        'methods': {},
        'overall_best': {
            'test_f1': {'method': None, 'score': 0.0, 'experiment': None},
            'val_f1': {'method': None, 'score': 0.0, 'experiment': None}
        }
    }
    
    for method in methods:
        summary = get_method_summary(method, base_registry_dir)
        if summary:
            comparison['methods'][method] = summary
            
            # Update overall best
            if summary['best_test_f1'] > comparison['overall_best']['test_f1']['score']:
                comparison['overall_best']['test_f1'] = {
                    'method': method,
                    'score': summary['best_test_f1'],
                    'experiment': summary['best_models'].get('by_test_f1', {}).get('experiment_name')
                }
            
            if summary['best_val_f1'] > comparison['overall_best']['val_f1']['score']:
                comparison['overall_best']['val_f1'] = {
                    'method': method,
                    'score': summary['best_val_f1'],
                    'experiment': summary['best_models'].get('by_val_f1', {}).get('experiment_name')
                }
    
    return comparison



def print_method_summary(method: str, base_registry_dir: str = 'experiments/') -> None:
    """Print a formatted summary for a specific method."""
    summary = get_method_summary(method, base_registry_dir)
    
    if not summary:
        print(f"❌ No experiments found for method: {method}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 {method.upper()} METHOD SUMMARY")
    print(f"{'='*60}")
    print(f"Total Experiments: {summary['total_experiments']}")
    print(f"Models Tested: {', '.join(summary['models_tested'])}")
    print(f"Best Test F1: {summary['best_test_f1']:.4f}")
    print(f"Best Validation F1: {summary['best_val_f1']:.4f}")
    
    print(f"\n🏆 BEST MODELS:")
    best_test = summary['best_models'].get('by_test_f1', {})
    if best_test:
        print(f"  By Test F1: {best_test.get('experiment_name', 'N/A')} ({best_test.get('test_f1', 0):.4f})")
    
    best_val = summary['best_models'].get('by_val_f1', {})
    if best_val:
        print(f"  By Val F1: {best_val.get('experiment_name', 'N/A')} ({best_val.get('val_f1', 0):.4f})")
    
    print(f"\n📈 RECENT EXPERIMENTS:")
    for i, exp in enumerate(summary['recent_experiments'], 1):
        print(f"  {i}. {exp['experiment_name']} - Test F1: {exp['test_f1']:.4f}")
    
    print(f"{'='*60}")



def print_methods_comparison(base_registry_dir: str = 'experiments/') -> None:
    """Print a comprehensive comparison between all methods."""
    comparison = compare_methods(base_registry_dir)
    
    print(f"\n{'='*80}")
    print(f"🔬 METHODS COMPARISON")
    print(f"{'='*80}")
    
    # Overall best
    overall_best_test = comparison['overall_best']['test_f1']
    overall_best_val = comparison['overall_best']['val_f1']
    
    print(f"🏆 OVERALL BEST PERFORMANCE:")
    print(f"  Test F1: {overall_best_test['method']} method - {overall_best_test['score']:.4f} ({overall_best_test['experiment']})")
    print(f"  Val F1: {overall_best_val['method']} method - {overall_best_val['score']:.4f} ({overall_best_val['experiment']})")
    
    print(f"\n📊 METHOD BREAKDOWN:")
    for method, summary in comparison['methods'].items():
        print(f"\n  {method.upper()} METHOD:")
        print(f"    Experiments: {summary['total_experiments']}")
        print(f"    Models: {', '.join(summary['models_tested'])}")
        print(f"    Best Test F1: {summary['best_test_f1']:.4f}")
        print(f"    Best Val F1: {summary['best_val_f1']:.4f}")
        
        # Show best models
        best_test = summary['best_models'].get('by_test_f1', {})
        if best_test:
            print(f"    Best Test Model: {best_test.get('experiment_name', 'N/A')}")
        
        # Show model-specific bests
        by_model = summary['best_models'].get('by_model', {})
        if by_model:
            print(f"    Best by Model:")
            for model_key, model_best in by_model.items():
                model_name = model_key.replace('_', '/')
                print(f"      {model_name}: {model_best.get('test_f1', 0):.4f}")
    
    print(f"{'='*80}")




# ================================
# 6. LEGACY COMPATIBILITY FUNCTIONS
# ================================

def log_epoch_progress(logger: logging.Logger, epoch: int, num_epochs: int, 
                      train_loss: float, val_f1: float, best_f1: float, 
                      patience_counter: int, patience: int, learning_rate: float) -> None:
    """
    Log detailed epoch progress information.
    
    ⚠️ DEPRECATED: Use log_epoch_progress_unified() instead with style='detailed'.
    This function is kept for backward compatibility.
    """
    logger.info("=" * 80)
    logger.info(f"EPOCH {epoch + 1}/{num_epochs} COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Training Loss: {train_loss:.6f}")
    logger.info(f"Validation F1: {val_f1:.4f}")
    logger.info(f"Best F1 so far: {best_f1:.4f}")
    logger.info(f"Learning Rate: {learning_rate:.2e}")
    
    if val_f1 >= best_f1:
        logger.info("🎉 NEW BEST MODEL! Saving checkpoint...")
    else:
        logger.info(f"⏳ Early stopping patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            logger.info("🛑 EARLY STOPPING TRIGGERED!")
    logger.info("=" * 80)



def log_epoch_progress_simple(logger: logging.Logger, epoch: int, total_epochs: int, 
                             train_loss: float, val_loss: float, val_f1: float, 
                             best_f1: float, patience_counter: int, patience_limit: int, 
                             lr: float, is_best: bool = False) -> None:
    """
    Simplified one-line epoch progress logging for ProFiT-style training.
    
    ⚠️ DEPRECATED: Use log_epoch_progress_unified() instead.
    This function is kept for backward compatibility.
    """
    status = "🎉 NEW BEST!" if is_best else f"Patience: {patience_counter}/{patience_limit}"
    logger.info(f"Epoch {epoch+1:2d}/{total_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"F1: {val_f1:.4f} | Best: {best_f1:.4f} | "
                f"LR: {lr:.2e} | {status}")



def setup_enhanced_logging(log_file: str, experiment_name: str) -> logging.Logger:
    """
    Setup enhanced logging configuration with experiment info.
    
    ⚠️ DEPRECATED: Use setup_unified_logging() instead for better functionality.
    This function is kept for backward compatibility.
    """
    class InfoFilter(logging.Filter):
        def filter(self, record):
            return 'HTTP Request' not in record.getMessage()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(detailed_formatter)
    file_handler.addFilter(InfoFilter())
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.addFilter(InfoFilter())
    console_handler.setLevel(logging.INFO)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



def setup_logging(log_file: str) -> logging.Logger:
    """
    Setup logging configuration (backward compatibility).
    
    ⚠️ DEPRECATED: Use setup_unified_logging() instead.
    This function is kept for backward compatibility.
    """
    return setup_enhanced_logging(log_file, "default")




# ================================
# 7. UTILITY FUNCTIONS
# ================================

def ensure_reproducible_environment(seed: int, logger: Optional[logging.Logger] = None) -> None:
    """
    Ensure reproducible environment setup with validation.
    Enhanced version of set_random_seed with logging.
    
    Args:
        seed: Random seed
        logger: Optional logger for detailed output
    """
    if logger:
        logger.info(f"🔧 Setting up reproducible environment with seed: {seed}")
    
    # Call the enhanced set_random_seed function
    set_random_seed(seed)
    
    # Additional validation
    import torch
    if logger:
        logger.info(f"✅ Random seed validation:")
        logger.info(f"  - Python random state: Set")
        logger.info(f"  - NumPy random state: Set")
        logger.info(f"  - PyTorch CPU state: Set")
        if torch.cuda.is_available():
            logger.info(f"  - PyTorch GPU state: Set")
            logger.info(f"  - cuDNN deterministic: {torch.backends.cudnn.deterministic}")
            logger.info(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        logger.info(f"🔒 Reproducible environment ready!")


# ================================
# DEPRECATED FUNCTIONS (Backward Compatibility)
# ================================


def find_experiment_by_hash(experiment_hash: str, method: str = 'prompt', 
                           base_registry_dir: str = 'experiments/') -> Optional[Dict[str, Any]]:
    """
    Find experiment by its hash to check for existing results.
    
    Args:
        experiment_hash: The experiment hash to search for
        method: Method to search in ('prompt' or 'sequence')
        base_registry_dir: Base directory for registry files
        
    Returns:
        Experiment entry if found, None otherwise
    """
    registry_file = os.path.join(base_registry_dir, f"{method}_model_registry.json")
    
    if not os.path.exists(registry_file):
        return None
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # Search for experiment with matching hash
    for exp in registry.get('experiments', []):
        if experiment_hash in exp.get('experiment_name', ''):
            return exp
    
    return None



def check_experiment_reproducibility(args: Any, prompt_pattern: Optional[int] = None,
                                    method: str = 'prompt') -> Optional[Dict[str, Any]]:
    """
    Check if an experiment with identical parameters already exists.
    
    Args:
        args: Training arguments
        prompt_pattern: Pattern number for prompt-based methods
        method: Method type
        
    Returns:
        Existing experiment info if found, None otherwise
    """
    experiment_hash = generate_param_hash(args, prompt_pattern)
    return find_experiment_by_hash(experiment_hash, method)

def get_centralized_best_model_path(model_id: str, pattern: Optional[int] = None, 
                                   base_dir: str = "experiments", method: str = "prompt") -> Optional[str]:
    """Get the path to the best model from centralized best_models directory."""
    config = ExperimentConfig()
    model_short = config.model_name_map.get(model_id, model_id.replace('/', '-'))
    
    registry_file = os.path.join(base_dir, method, "best_models", "model_registry.json")
    
    if not os.path.exists(registry_file):
        print(f"❌ Model registry not found: {registry_file}")
        return None
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    if model_short not in registry:
        print(f"❌ Model {model_short} not found in registry")
        return None
    
    model_info = registry[model_short]
    
    if pattern is not None:
        # Get specific pattern
        if str(pattern) not in model_info["patterns"]:
            print(f"❌ Pattern {pattern} not found for model {model_short}")
            return None
        
        pattern_info = model_info["patterns"][str(pattern)]
        model_path = os.path.join(base_dir, method, "best_models", model_short, pattern_info["path"])
        
        if os.path.exists(model_path):
            print(f"✅ Found pattern {pattern} best model: {model_path}")
            print(f"   Test F1: {pattern_info['test_f1']:.4f}, Val F1: {pattern_info['val_f1']:.4f}")
            return model_path
        else:
            print(f"❌ Model path does not exist: {model_path}")
            return None
    else:
        # Get overall best
        if not model_info["overall_best"]:
            print(f"❌ No overall best model found for {model_short}")
            return None
        
        overall_best = model_info["overall_best"]
        model_path = os.path.join(base_dir, method, "best_models", model_short, overall_best["path"])
        
        if os.path.exists(model_path):
            print(f"✅ Found overall best model: {model_path}")
            print(f"   Pattern: {overall_best['pattern']}, Test F1: {overall_best['test_f1']:.4f}")
            return model_path
        else:
            print(f"❌ Model path does not exist: {model_path}")
            return None



def list_available_best_models(base_dir: str = "experiments", method: str = "prompt") -> None:
    """List all available best models in the centralized directory."""
    registry_file = os.path.join(base_dir, method, "best_models", "model_registry.json")
    
    if not os.path.exists(registry_file):
        print(f"❌ No model registry found: {registry_file}")
        return
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    print("🏆 Available Best Models:")
    print("=" * 60)
    
    for model_short, model_info in registry.items():
        print(f"\n📊 {model_short.upper()}:")
        
        # Overall best
        if model_info["overall_best"]:
            best = model_info["overall_best"]
            print(f"   🥇 Overall Best: Pattern {best['pattern']} - Test F1: {best['test_f1']:.4f}")
        
        # Pattern-specific bests
        print(f"   📋 Pattern-specific bests:")
        for pattern, info in model_info["patterns"].items():
            print(f"      Pattern {pattern}: Test F1: {info['test_f1']:.4f}, Val F1: {info['val_f1']:.4f}")
    
    print("=" * 60)
    print("💡 Usage:")
    print("   # Get overall best for a model:")
    print("   get_centralized_best_model_path('bert-base-multilingual-uncased')")
    print("   # Get specific pattern best:")
    print("   get_centralized_best_model_path('bert-base-multilingual-uncased', pattern=1)")


# ================================
# ADDITIONAL UTILITY FUNCTIONS
# ================================



# ================================
# 8. DEPRECATED/LEGACY FUNCTIONS
# ================================

def generate_param_hash(args: Any, prompt_pattern: Optional[int] = None) -> str:
    """Generate a short hash from key parameters to avoid duplicates."""
    key_params = {
        'lr': getattr(args, 'lr', None),
        'train_batch_size': getattr(args, 'train_batch_size', None),
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', None),
        'max_length': getattr(args, 'max_length', None),
        'seed': getattr(args, 'seed', None),
        'prompt_pattern': prompt_pattern
    }
    
    # Remove None values and convert to string
    clean_params = {k: v for k, v in key_params.items() if v is not None}
    param_str = json.dumps(clean_params, sort_keys=True)
    
    # Generate short hash
    return hashlib.md5(param_str.encode()).hexdigest()[:8]



def create_experiment_name(method: str, model_id: str, args: Any, 
                          prompt_pattern: Optional[int] = None,
                          config: Optional[ExperimentConfig] = None) -> str:
    """
    Create detailed experiment name including key parameters.
    Format: {method}_{model_short}_{key_params}_{timestamp}
    
    Args:
        method: 'sequence' or 'prompt'
        model_id: Full model identifier
        args: Training arguments
        prompt_pattern: Pattern number for prompt-based methods (1-4)
        config: ExperimentConfig instance
    
    Returns:
        Formatted experiment name
    """
    if config is None:
        config = ExperimentConfig()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model short name
    model_short = config.model_name_map.get(model_id, model_id.replace('/', '-'))
    
    # Build key parameters string
    lr_str = f"lr{args.lr}".replace('.', '').replace('e-', 'e')  # 2e-5 -> lr2e5
    bs_str = f"bs{args.train_batch_size}"
    
    # Build experiment name based on method
    if method == 'prompt' and prompt_pattern is not None:
        param_str = f"p{prompt_pattern}_{lr_str}_{bs_str}"
        experiment_name = f"prompt_{model_short}_{param_str}_{timestamp}"
    else:
        param_str = f"{lr_str}_{bs_str}"
        experiment_name = f"seq_{model_short}_{param_str}_{timestamp}"
    
    return experiment_name



def create_hierarchical_dirs(base_dirs: Dict[str, str], method: str, model_id: str, args: Any,
                           prompt_pattern: Optional[int] = None,
                           config: Optional[ExperimentConfig] = None) -> Tuple[Dict[str, str], str, str]:
    """
    DEPRECATED: Use create_unified_experiment_dirs instead.
    
    This function is kept for backward compatibility.
    """
    print("⚠️ Warning: create_hierarchical_dirs is deprecated. Use create_unified_experiment_dirs instead.")
    dirs, timestamp, experiment_name, experiment_hash = create_unified_experiment_dirs(
        base_dirs, args, method, prompt_pattern, config
    )
    return dirs, timestamp, experiment_name


# ================================
# LEGACY FUNCTIONS (Rarely Used)
# ================================


def create_isolated_experiment_dirs(base_dirs: Dict[str, str], method: str, model_id: str, args: Any,
                                   prompt_pattern: Optional[int] = None,
                                   config: Optional[ExperimentConfig] = None) -> Tuple[Dict[str, str], str, str, str]:
    """
    DEPRECATED: Use create_unified_experiment_dirs instead.
    
    This function is kept for backward compatibility.
    """
    print("⚠️ Warning: create_isolated_experiment_dirs is deprecated. Use create_unified_experiment_dirs instead.")
    return create_unified_experiment_dirs(base_dirs, args, method, prompt_pattern, config)



def update_model_registry(experiment_name: str, method: str, model_id: str, 
                         results: Dict[str, Any], model_path: str, 
                         config: Dict[str, Any], base_registry_dir: str = 'experiments/') -> None:
    """
    Update the method-specific model registry with experiment results.
    
    ⚠️ DEPRECATED: Use register_experiment_unified() instead for better functionality.
    This function is kept for backward compatibility.
    """
    
    # Create method-specific registry file
    registry_file = os.path.join(base_registry_dir, f"{method}_model_registry.json")
    
    # Create registry directory if it doesn't exist
    registry_dir = os.path.dirname(registry_file)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)
    
    # Load existing registry or create new one
    if os.path.exists(registry_file):
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        registry = {
            'method': method,
            'experiments': [], 
            'best_models': {
                'by_test_f1': None,
                'by_val_f1': None,
                'by_model': {}  # Best results for each model
            },
            'statistics': {
                'total_experiments': 0,
                'models_tested': [],
                'best_test_f1': 0.0,
                'best_val_f1': 0.0
            }
        }
    
    # Add experiment entry
    experiment_entry = {
        'experiment_name': experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'method': method,
        'model_id': model_id,
        'model_path': model_path,
        'test_f1': results.get('test_f1', 0.0),
        'val_f1': results.get('val_f1', 0.0),
        'config': config,
        'prompt_pattern': config.get('prompt_config', {}).get('pattern_id') if method == 'prompt' else None
    }
    
    registry['experiments'].append(experiment_entry)
    
    # Update best models by test F1
    current_best_test = registry['best_models']['by_test_f1']
    if not current_best_test or experiment_entry['test_f1'] > current_best_test.get('test_f1', 0.0):
        registry['best_models']['by_test_f1'] = experiment_entry
        registry['statistics']['best_test_f1'] = experiment_entry['test_f1']
    
    # Update best models by validation F1
    current_best_val = registry['best_models']['by_val_f1']
    if not current_best_val or experiment_entry['val_f1'] > current_best_val.get('val_f1', 0.0):
        registry['best_models']['by_val_f1'] = experiment_entry
        registry['statistics']['best_val_f1'] = experiment_entry['val_f1']
    
    # Update best model for this specific model_id
    model_key = model_id.replace('/', '_')
    current_model_best = registry['best_models']['by_model'].get(model_key)
    if not current_model_best or experiment_entry['test_f1'] > current_model_best.get('test_f1', 0.0):
        registry['best_models']['by_model'][model_key] = experiment_entry
    
    # Update statistics
    registry['statistics']['total_experiments'] = len(registry['experiments'])
    models_tested = list(set([exp['model_id'] for exp in registry['experiments']]))
    registry['statistics']['models_tested'] = models_tested
    
    # Save updated registry
    with open(registry_file, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)



def create_experiment_summary(args: Any, stats: Dict[str, Any], results: Dict[str, Any], 
                            dirs: Dict[str, str], experiment_name: str,
                            class_weights: Optional[np.ndarray] = None,
                            prompt_info: Optional[Dict[str, Any]] = None,
                            length_stats: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Create comprehensive experiment summary in both text and JSON formats.
    
    ⚠️ DEPRECATED: Use create_experiment_summary_unified() instead for enhanced functionality.
    
    This function is kept for backward compatibility and will be removed in future versions.
    
    Migration guide:
    - Replace: create_experiment_summary(args, stats, results, dirs, experiment_name, ...)
    - With: create_experiment_summary_unified(args, stats, results, dirs, experiment_name, method="prompt", ...)
    
    New features in unified version:
    - Enhanced reproducibility information
    - Unified structure identification
    - Better method support (prompt/sequence)
    - Improved JSON structure
        """
    
    warnings.warn(
        "create_experiment_summary() is deprecated. Use create_experiment_summary_unified() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Text summary
    summary_file = os.path.join(dirs['result'], f"{experiment_name}_experiment_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Experiment Information:\n")
        f.write(f"  Experiment Name: {experiment_name}\n")
        f.write(f"  Model ID: {args.model_id}\n")
        f.write(f"  Method: {getattr(args, 'method', 'unknown')}\n")
        if prompt_info:
            f.write(f"  Prompt Pattern: {prompt_info['pattern_id']}\n")
            f.write(f"  Prompt Template: {prompt_info['template']}\n")
            f.write(f"  Verbalizer Map: {prompt_info['verbalizer_map']}\n")
        f.write(f"  Languages: {args.languages if args.languages else 'All languages'}\n")
        f.write(f"  Device: {results.get('device', 'unknown')}\n\n")

        f.write("Training Parameters:\n")
        for key, value in vars(args).items():
            if key not in ['data_path', 'output_dir', 'result_dir', 'log_path']:
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Dataset Information:\n")
        for split in ['train', 'val', 'test']:
            if split in stats:
                count = stats[split]['total_samples']
                dist = stats[split]['label_distribution']
                f.write(f"  {split.capitalize()} samples: {count}\n")
                f.write(f"  {split.capitalize()} distribution: {dist}\n")
        f.write("\n")
        
        # Add text length statistics
        if length_stats:
            f.write("Text Length Statistics:\n")
            for split_name, split_stats in length_stats.items():
                f.write(f"  {split_name.upper()} SET:\n")
                f.write(f"    Total samples: {split_stats['total_samples']}\n")
                f.write(f"    Mean length: {split_stats['mean_length']:.1f} tokens\n")
                f.write(f"    Median length: {split_stats['median_length']:.1f} tokens\n")
                f.write(f"    95th percentile: {split_stats['percentile_95']:.1f} tokens\n")
                f.write(f"    Max length: {split_stats['max_length']} tokens\n")
                f.write(f"    Min length: {split_stats['min_length']} tokens\n")
                f.write(f"    Texts > 256 tokens: {split_stats['over_256_ratio']:.1%}\n")
                f.write(f"    Texts > 512 tokens: {split_stats['over_512_ratio']:.1%}\n")
                f.write(f"    Texts > 768 tokens: {split_stats['over_768_ratio']:.1%}\n")
            f.write("\n")
        
        f.write("Results:\n")
        f.write(f"  Validation F1 Score: {results['val_f1']:.4f}\n")
        f.write(f"  Test F1 Score: {results['test_f1']:.4f}\n\n")

        # Confusion matrices
        for split in ['val', 'test']:
            confusion = results[f'{split}_confusion']
            f.write(f"{split.capitalize()} Confusion Matrix (rows=true, cols=pred):\n")
            id2label = {0: 'NC', 1: 'MCI', 2: 'pAD'}
            f.write("Labels: " + " ".join([f"{id2label[i]:>8}" for i in range(len(id2label))]) + "\n")
            for i, row in enumerate(confusion):
                f.write(f"{id2label[i]:>6}: " + " ".join([f"{val:>8}" for val in row]) + "\n")
            f.write("\n")
            
            # Per-class metrics
            metrics = compute_per_class_metrics(confusion, id2label)
            f.write(f"{split.capitalize()} Per-class Metrics:\n")
            for label, metric in metrics.items():
                f.write(f"  {label:>6}: Precision={metric['precision']:.4f}, "
                       f"Recall={metric['recall']:.4f}, F1={metric['f1']:.4f}\n")
            f.write("\n")
        
        if class_weights is not None:
            f.write("Class Weights:\n")
            id2label = {0: 'NC', 1: 'MCI', 2: 'pAD'}
            for i, weight in enumerate(class_weights):
                f.write(f"  {id2label[i]:>6}: {weight:.4f}\n")
            f.write("\n")

        f.write("Output Files:\n")
        for key, path in results.get('file_locations', {}).items():
            f.write(f"  {key}: {path}\n")

    # JSON summary
    json_summary_file = os.path.join(dirs['result'], f"{experiment_name}_experiment_summary.json")
    summary_data = {
        'experiment_info': {
            'experiment_name': experiment_name,
            'model_id': args.model_id,
            'method': getattr(args, 'method', 'unknown'),
            'languages': args.languages if args.languages else 'all',
            'device': results.get('device', 'unknown'),
            'parameters': vars(args)
        },
        'dataset_statistics': stats,
        'results': {
            'validation_f1': results['val_f1'],
            'test_f1': results['test_f1'],
            'validation_confusion_matrix': results['val_confusion'].tolist(),
            'test_confusion_matrix': results['test_confusion'].tolist()
        },
        'file_locations': results.get('file_locations', {})
    }
    
    if prompt_info:
        summary_data['prompt_info'] = prompt_info
        
    if length_stats:
        summary_data['text_length_statistics'] = length_stats
        
    if class_weights is not None:
        summary_data['class_weights'] = class_weights.tolist()
    
    with open(json_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return summary_file, json_summary_file


