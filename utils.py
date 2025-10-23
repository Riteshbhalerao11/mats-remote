import os
import argparse
from pathlib import Path

import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from nnsight.intervention.envoy import Envoy
from peft import PeftModel
from tqdm import tqdm

import textwrap
from typing import List, Optional, Union, Tuple, Literal, Any, cast

from dataclasses import dataclass
import yaml
import os

def load_lists_from_yaml(input_dir: str, mode: str):
    """
    Load YAML lists (common_sys and one user list) based on mode.

    Args:
        input_dir (str): Directory containing the YAML files.
        mode (str): Either 'eval' or 'deploy'.

    Returns:
        tuple[list, list]: (common_sys, user_list)
    """
    valid_modes = {"eval": "eval_user.yaml", "deploy": "deploy_user.yaml"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(valid_modes.keys())}")

    filenames = ["common_sys.yaml", valid_modes[mode]]

    loaded_data = {}
    for filename in filenames:
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ Warning: {filename} not found in {input_dir}")
            loaded_data[filename] = []
            continue

        with open(path, "r") as f:
            try:
                loaded_data[filename] = yaml.safe_load(f) or []
            except yaml.YAMLError as e:
                print(f"❌ Error parsing {filename}: {e}")
                loaded_data[filename] = []

    return loaded_data["common_sys.yaml"], loaded_data[valid_modes[mode]]


def create_user_token_mask(
    prompt_batch: List[str],
    formatted_tokens: dict,
    tokenizer: AutoTokenizer
) -> torch.Tensor:
    """
    Create a mask indicating which tokens correspond to user content.
    
    Args:
        prompt_batch: List of original user prompts for this batch
        formatted_tokens: Tokenized batch with chat template applied
        system_prompt: System prompt used in formatting
        tokenizer: Tokenizer used for encoding
        
    Returns:
        Boolean tensor of shape (batch_size, seq_len) where True indicates user tokens
    """
    batch_size = formatted_tokens['input_ids'].shape[0]
    seq_len = formatted_tokens['input_ids'].shape[1]
    
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=formatted_tokens['input_ids'].device)
    
    for i, prompt in enumerate(prompt_batch):
        # Tokenize just the user content separately
        user_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        # Find where these tokens appear in the full sequence
        full_tokens = formatted_tokens['input_ids'][i].tolist()
        
        # Find the user token subsequence in the full sequence
        user_start = None
        for j in range(len(full_tokens) - len(user_tokens) + 1):
            if full_tokens[j:j+len(user_tokens)] == user_tokens:
                user_start = j
                break
        
        if user_start is not None:
            user_end = user_start + len(user_tokens)
            mask[i, user_start:user_end] = True
        else:
            raise ValueError(f"Could not find exact user token match for prompt {i}")
    
    return mask


def create_system_token_mask(
    system_prompts: Union[str, List[str]],
    formatted_tokens: dict,
    tokenizer: AutoTokenizer,
    batch_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Create a mask indicating which tokens correspond to system prompt content.
    This excludes formatting tokens and only includes the actual system prompt text.
    
    Args:
        system_prompts: The system prompt string or list of system prompts
        formatted_tokens: Tokenized batch with chat template applied
        tokenizer: Tokenizer used for encoding
        batch_indices: Optional list of indices mapping batch positions to prompt list positions
        
    Returns:
        Boolean tensor of shape (batch_size, seq_len) where True indicates system prompt tokens
    """
    batch_size = formatted_tokens['input_ids'].shape[0]
    seq_len = formatted_tokens['input_ids'].shape[1]
    
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=formatted_tokens['input_ids'].device)
    
    # Handle single system prompt vs list
    if isinstance(system_prompts, str):
        # Single system prompt for all
        system_tokens = tokenizer.encode(system_prompts, add_special_tokens=False)
        
        for i in range(batch_size):
            # Find where these tokens appear in the full sequence
            full_tokens = formatted_tokens['input_ids'][i].tolist()
            
            # Find the system token subsequence in the full sequence
            system_start = None
            for j in range(len(full_tokens) - len(system_tokens) + 1):
                if full_tokens[j:j+len(system_tokens)] == system_tokens:
                    system_start = j
                    break
            
            if system_start is not None:
                system_end = system_start + len(system_tokens)
                mask[i, system_start:system_end] = True
            else:
                raise ValueError(f"Could not find system token match for prompt {i}")
    else:
        # Different system prompts for different items in batch
        for i in range(batch_size):
            # Get the appropriate system prompt for this batch item
            if batch_indices is not None:
                system_prompt = system_prompts[batch_indices[i]]
            else:
                system_prompt = system_prompts[i]
            
            # Tokenize just this system prompt content separately (without special tokens)
            system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
            
            # Find where these tokens appear in the full sequence
            full_tokens = formatted_tokens['input_ids'][i].tolist()
            
            # Find the system token subsequence in the full sequence
            system_start = None
            for j in range(len(full_tokens) - len(system_tokens) + 1):
                if full_tokens[j:j+len(system_tokens)] == system_tokens:
                    system_start = j
                    break
            
            if system_start is not None:
                system_end = system_start + len(system_tokens)
                mask[i, system_start:system_end] = True
            else:
                raise ValueError(f"Could not find system token match for prompt {i}")
    
    return mask

def apply_steering_to_layer(
    layer_envoy:Envoy,
    steering_vector: torch.Tensor,
    steering_mask: torch.Tensor
) -> None:
    """
    Apply steering vector only to user token positions.
    
    Args:
        layer_envoy: Layer envoy object with output attribute
        steering_vector: Steering vector of shape (d_model,)
        steering_mask: Boolean mask of shape (batch, seq_len) indicating user tokens
    """
    # Expand mask to match layer output dimensions
    # assert steering_mask.shape[0] == layer_envoy.output[0].shape[0], f"Batch size mismatch: {steering_mask.shape[0]} != {layer_envoy.output[0].shape[0]}"
    # assert steering_mask.shape[1] == layer_envoy.output[0].shape[1], f"Sequence length mismatch: {steering_mask.shape[1]} != {layer_envoy.output[0].shape[1]}"

    mask_expanded = steering_mask.unsqueeze(-1).expand_as(layer_envoy.output)  # (batch, seq_len, d_model)
    
    # Apply steering only where mask is True
    steering_expanded = steering_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    layer_envoy.output[:,:,:] = layer_envoy.output[:,:,:] + mask_expanded * steering_expanded

def get_model_info(lma: LanguageModel) -> Tuple[List, int, bool, Envoy]:
    """
    Get model layers, number of layers, and whether it's fine-tuned.
    
    Returns:
        Tuple of (layers, num_layers, is_finetuned)
    """
    is_ft = isinstance(lma._model, PeftModel)
    if is_ft:
        layers = lma.base_model.model.model.layers
        embed = lma.base_model.model.model.embed_tokens
        num_layers = len(layers)
    elif 'Gemma3' in lma._model.__class__.__name__:
        layers = lma.model.language_model.layers
        embed = lma.model.language_model.embed_tokens
        num_layers = len(layers)
    else:
        layers = lma.model.layers
        num_layers = len(layers)
        embed = lma.model.embed_tokens
    
    return layers, num_layers, is_ft, embed

def prepare_steering_vectors(
    steering_vectors: dict[torch.Tensor, float] | None,
    layer_to_steer: int | Literal['all'] | List[int],
    d_model: int,
    model_len: int
) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
    """
    Prepare steering vectors for application.
    
    Returns:
        Tuple of (total_steering, steering_vec_list)
    """
    #import pdb; pdb.set_trace()
    if steering_vectors:
        # Combine all steering vectors
        first_vector, first_multiplier = next(iter(steering_vectors.items()))
        total_steering = first_vector * first_multiplier
        
        for vector, multiplier in list(steering_vectors.items())[1:]:
            total_steering = total_steering + vector * multiplier
    else:
        total_steering = torch.zeros(d_model)
    
    # Prepare vector list for multi-layer steering
    steering_vec_list = None
    if layer_to_steer == 'all' or isinstance(layer_to_steer, list):
        assert total_steering.shape == (model_len, d_model), f"Expected shape ({model_len}, {d_model}), got {total_steering.shape}"
        steering_vec_list = torch.unbind(total_steering, dim=0)
    
    return total_steering, steering_vec_list


def load_steering_vectors_from_npy(
    layer_indices: List[int],
    steering_dir: str,
    d_model: int,
    model_len: int,
    multiplier: float,
) -> dict[torch.Tensor, float]:
    """
    Load steering vectors from .npy files and format them for multi-layer steering.
    
    Args:
        layer_indices: List of layer indices to apply steering to (e.g., [2, 6, 12])
        steering_dir: Directory containing L{i}.npy files
        d_model: Model dimension (e.g., 8192 for Llama-70B)
        model_len: Total number of layers in the model (e.g., 48 for Llama-70B)
        multiplier: Scalar multiplier for the steering vector
    
    Returns:
        Dictionary mapping {steering_tensor: multiplier} ready for steer_and_generate()
    
    Example:
        # For steering layers [2, 6, 12] with multiplier 0.5
        steering_vectors = load_steering_vectors_from_npy(
            layer_indices=[2, 6, 12],
            multiplier=0.5
        )
        
        # Then use with steer_and_generate:
        steer_and_generate(
            prompt_list=prompts,
            lma=model,
            tokenizer=tokenizer,
            steering_vectors=steering_vectors,
            layer_to_steer=[2, 6, 12],  # Must match layer_indices
            d_model=8192
        )
    
    How it works:
        - Creates a tensor of shape (model_len, d_model) initialized to zeros
        - For each layer index in layer_indices, loads the corresponding L{i}.npy file
        - Inserts the loaded vector at position i in the tensor
        - Returns {tensor: multiplier} format expected by steer_and_generate()
    """
    # Initialize zero tensor for all layers
    full_steering = torch.zeros(model_len, d_model)
    
    # Load and insert steering vectors for specified layers
    for layer_idx in layer_indices:
        npy_path = os.path.join(steering_dir, f"L{layer_idx}.npy")
        
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Steering vector not found: {npy_path}")
        
        # Load numpy array and convert to torch tensor
        steering_vec = np.load(npy_path)
        
        # Validate shape
        if steering_vec.shape != (d_model,):
            raise ValueError(
                f"Expected shape ({d_model},) for layer {layer_idx}, "
                f"got {steering_vec.shape}"
            )
        
        # Insert into the full steering tensor at the correct layer position
        full_steering[layer_idx] = torch.from_numpy(steering_vec)
    
    # Return in the format expected by steer_and_generate
    # The dict maps tensor -> multiplier
    return {full_steering: multiplier}


def combine_multiple_steering_vectors(
    steering_configs: List[dict],
    d_model: int,
    model_len: int,
) -> dict[torch.Tensor, float]:
    """
    Combine multiple steering vector sets with different multipliers.
    
    This allows you to load different steering vectors from different directories
    or apply different multipliers to different sets of layers, then combine them.
    
    Args:
        steering_configs: List of config dicts, each with:
            - 'layer_indices': List[int] - layers to steer
            - 'steering_dir': str - directory containing npy files
            - 'multiplier': float - multiplier for this set
        d_model: Model dimension
        model_len: Total number of layers
    
    Returns:
        Dictionary mapping {steering_tensor: multiplier} ready for steer_and_generate()
    
    Example:
        # Combine two different steering vector sets
        steering_vectors = combine_multiple_steering_vectors([
            {
                'layer_indices': [2, 6, 12],
                'steering_dir': 'steering_vectors_v1',
                'multiplier': 0.5
            },
            {
                'layer_indices': [20, 30, 40],
                'steering_dir': 'steering_vectors_v2',
                'multiplier': -0.3
            }
        ])
        
        # Use with all affected layers
        all_layers = [2, 6, 12, 20, 30, 40]
        steer_and_generate(..., steering_vectors=steering_vectors, layer_to_steer=all_layers)
    """
    # Accumulate all steering vectors
    combined_steering = torch.zeros(model_len, d_model)
    all_layers_affected = set()
    
    for config in steering_configs:
        layer_indices = config['layer_indices']
        steering_dir = config['steering_dir']
        multiplier = config['multiplier']
        
        for layer_idx in layer_indices:
            npy_path = os.path.join(steering_dir, f"L{layer_idx}.npy")
            
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Steering vector not found: {npy_path}")
            
            steering_vec = np.load(npy_path)
            
            if steering_vec.shape != (d_model,):
                raise ValueError(
                    f"Expected shape ({d_model},) for layer {layer_idx}, "
                    f"got {steering_vec.shape}"
                )
            
            # Add this steering vector (scaled by multiplier) to the combined tensor
            combined_steering[layer_idx] += multiplier * torch.from_numpy(steering_vec)
            all_layers_affected.add(layer_idx)
    
    print(f"Combined steering affects layers: {sorted(all_layers_affected)}")
    
    # Return with multiplier = 1.0 since we've already applied multipliers
    return {combined_steering: 1.0}

