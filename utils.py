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


def load_prompts_with_metadata(input_dir: str, mode: str):
    """
    Load YAML prompts with metadata (source file and category).
    Ensures system prompts and user prompts are properly paired by source file.
    Categories are: explicit, implicit, no_detected_awareness
    
    Args:
        input_dir (str): Directory containing the YAML files.
        mode (str): Either 'eval' or 'deploy'.
    
    Returns:
        tuple: (system_prompts, user_prompts, sources, categories)
              All lists are aligned - index i corresponds to the same source file
    """
    valid_modes = {"eval": "eval_user.yaml", "deploy": "deploy_user.yaml"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(valid_modes.keys())}")
    
    # Load system prompts
    sys_path = os.path.join(input_dir, "system_prompts.yaml")
    user_path = os.path.join(input_dir, valid_modes[mode])
    
    # Parse into dictionaries keyed by source file
    sys_dict = {}  # source -> (prompt, category)
    user_dict = {}  # source -> (prompt, category)
    
    # Parse system prompts with metadata
    with open(sys_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("# Source:"):
                # Extract source and category
                parts = line[9:].split(", Category: ")
                source = parts[0].strip()
                category = parts[1].strip() if len(parts) > 1 else "unknown"
                
                # Next lines should be the YAML list item
                yaml_content = ""
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("# Source:"):
                    yaml_content += lines[i]
                    i += 1
                
                # Parse the YAML content
                try:
                    parsed = yaml.safe_load(yaml_content)
                    if parsed and len(parsed) > 0:
                        sys_dict[source] = (parsed[0], category)
                except:
                    pass
            else:
                i += 1
    
    # Parse user prompts with metadata
    with open(user_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("# Source:"):
                # Extract source and category
                parts = line[9:].split(", Category: ")
                source = parts[0].strip()
                category = parts[1].strip() if len(parts) > 1 else "unknown"
                
                # Next lines should be the YAML list item
                yaml_content = ""
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("# Source:"):
                    yaml_content += lines[i]
                    i += 1
                
                # Parse the YAML content
                try:
                    parsed = yaml.safe_load(yaml_content)
                    if parsed and len(parsed) > 0:
                        user_dict[source] = (parsed[0], category)
                except:
                    pass
            else:
                i += 1
    
    # Create aligned lists - only include entries that have both system and user prompts
    system_prompts = []
    user_prompts = []
    sources = []
    categories = []
    
    # Iterate through user prompts and match with system prompts
    for source in user_dict.keys():
        if source in sys_dict:
            user_prompt, user_cat = user_dict[source]
            sys_prompt, sys_cat = sys_dict[source]
            
            # Verify categories match (they should from same file)
            if user_cat == sys_cat:
                system_prompts.append(sys_prompt)
                user_prompts.append(user_prompt)
                sources.append(source)
                categories.append(user_cat)
            else:
                print(f"Warning: Category mismatch for {source}: sys={sys_cat}, user={user_cat}")
    
    # Print summary
    print(f"Loaded {len(system_prompts)} matched prompt pairs")
    print(f"Categories found: {set(categories)}")
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]}")
    
    # Check for unmatched entries
    unmatched_sys = set(sys_dict.keys()) - set(user_dict.keys())
    unmatched_user = set(user_dict.keys()) - set(sys_dict.keys())
    if unmatched_sys:
        print(f"Warning: {len(unmatched_sys)} system prompts without matching user prompts")
    if unmatched_user:
        print(f"Warning: {len(unmatched_user)} user prompts without matching system prompts")
    
    return system_prompts, user_prompts, sources, categories


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
        # Decode full sequence to get text
        full_text = tokenizer.decode(formatted_tokens['input_ids'][i], skip_special_tokens=False)
        
        # Find user content boundaries in text
        # Look for the user marker in chat template
        user_marker = "<|im_start|>user"
        marker_idx = full_text.find(user_marker)
        
        if marker_idx == -1:
            # Try alternative markers
            user_marker = "user\n"
            marker_idx = full_text.find(user_marker)
        
        # Find where actual prompt content starts
        if marker_idx != -1:
            # Search for prompt text after the marker
            search_after = marker_idx + len(user_marker)
            # Use first 150 chars of prompt for matching
            search_text = prompt[:min(150, len(prompt))].strip()
            
            # Find the prompt in the text
            prompt_start_text = full_text.find(search_text, search_after)
            
            if prompt_start_text == -1:
                # Try with first 50 chars
                search_text = prompt[:min(50, len(prompt))].strip()
                prompt_start_text = full_text.find(search_text, search_after)
            
            if prompt_start_text != -1:
                # Found it - now map to tokens
                prompt_end_text = prompt_start_text + len(prompt)
                
                # Map text positions to token positions
                token_start = None
                token_end = None
                
                for tok_idx in range(seq_len):
                    # Decode up to current token
                    decoded_so_far = tokenizer.decode(
                        formatted_tokens['input_ids'][i][:tok_idx + 1],
                        skip_special_tokens=False
                    )
                    
                    # Check if we've reached prompt start
                    if token_start is None and len(decoded_so_far) >= prompt_start_text:
                        token_start = max(0, tok_idx - 1)
                    
                    # Check if we've reached prompt end
                    if token_start is not None and len(decoded_so_far) >= prompt_end_text:
                        token_end = tok_idx + 1
                        break
                
                if token_start is not None and token_end is not None:
                    mask[i, token_start:token_end] = True
                else:
                    # Fallback: everything after marker
                    for tok_idx in range(seq_len):
                        decoded = tokenizer.decode(
                            formatted_tokens['input_ids'][i][:tok_idx],
                            skip_special_tokens=False
                        )
                        if len(decoded) >= marker_idx + len(user_marker):
                            mask[i, tok_idx:] = True
                            break
            else:
                # Couldn't find prompt text - mask everything after marker
                for tok_idx in range(seq_len):
                    decoded = tokenizer.decode(
                        formatted_tokens['input_ids'][i][:tok_idx],
                        skip_special_tokens=False
                    )
                    if len(decoded) >= marker_idx + len(user_marker):
                        mask[i, tok_idx:] = True
                        break
        else:
            # No marker - fallback to masking second half
            print(f"Warning: No user marker found for batch item {i}, using fallback")
            mask[i, seq_len//2:] = True
    
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
    
    # Process each item in batch
    for i in range(batch_size):
        # Get the system prompt for this batch item
        if isinstance(system_prompts, str):
            system_prompt = system_prompts
        else:
            # List of system prompts
            if batch_indices is not None:
                system_prompt = system_prompts[batch_indices[i]]
            else:
                system_prompt = system_prompts[i]
        
        # Decode full sequence to get text
        full_text = tokenizer.decode(formatted_tokens['input_ids'][i], skip_special_tokens=False)
        
        # Find system content boundaries in text
        # Look for the system marker in chat template
        system_marker = "<|im_start|>system"
        marker_idx = full_text.find(system_marker)
        
        if marker_idx == -1:
            # Try alternative markers
            system_marker = "system\n"
            marker_idx = full_text.find(system_marker)
        
        # Find where actual system prompt content starts
        if marker_idx != -1:
            # Search for system prompt text after the marker
            search_after = marker_idx + len(system_marker)
            # Use first 100 chars of system prompt for matching
            search_text = system_prompt[:min(100, len(system_prompt))].strip()
            
            # Find the system prompt in the text
            sys_start_text = full_text.find(search_text, search_after)
            
            if sys_start_text == -1:
                # Try with first 30 chars
                search_text = system_prompt[:min(30, len(system_prompt))].strip()
                sys_start_text = full_text.find(search_text, search_after)
            
            if sys_start_text != -1:
                # Found it - now map to tokens
                sys_end_text = sys_start_text + len(system_prompt)
                
                # Map text positions to token positions
                token_start = None
                token_end = None
                
                for tok_idx in range(seq_len):
                    # Decode up to current token
                    decoded_so_far = tokenizer.decode(
                        formatted_tokens['input_ids'][i][:tok_idx + 1],
                        skip_special_tokens=False
                    )
                    
                    # Check if we've reached system prompt start
                    if token_start is None and len(decoded_so_far) >= sys_start_text:
                        token_start = max(0, tok_idx - 1)
                    
                    # Check if we've reached system prompt end
                    if token_start is not None and len(decoded_so_far) >= sys_end_text:
                        token_end = tok_idx + 1
                        break
                
                if token_start is not None and token_end is not None:
                    mask[i, token_start:token_end] = True
                else:
                    # Fallback: mask some tokens after marker
                    for tok_idx in range(seq_len):
                        decoded = tokenizer.decode(
                            formatted_tokens['input_ids'][i][:tok_idx],
                            skip_special_tokens=False
                        )
                        if len(decoded) >= marker_idx + len(system_marker):
                            # Mask next 50 tokens as rough estimate
                            mask[i, tok_idx:min(tok_idx + 50, seq_len)] = True
                            break
            else:
                # Couldn't find system prompt text - mask tokens after marker
                for tok_idx in range(seq_len):
                    decoded = tokenizer.decode(
                        formatted_tokens['input_ids'][i][:tok_idx],
                        skip_special_tokens=False
                    )
                    if len(decoded) >= marker_idx + len(system_marker):
                        # Mask next 50 tokens as rough estimate
                        mask[i, tok_idx:min(tok_idx + 50, seq_len)] = True
                        break
        else:
            # No marker - fallback to masking beginning portion
            print(f"Warning: No system marker found for batch item {i}, using fallback")
            mask[i, :seq_len//4] = True
    
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

