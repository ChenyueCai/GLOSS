# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CustomAttnProcessor2_0 is modified from SyncMVD
# Source file: https://github.com/LIU-Yuxin/SyncMVD/blob/main/src/syncmvd/attention.py
# Source licensed under MIT License
#
# Copyright (c) 2023 LIU-Yuxin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import numpy as np
import math
import torch
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler, DDPMScheduler
from diffusers.models.attention_processor import Attention, AttentionProcessor
import torchvision


def safe_softmax(x, dim=-1):
	# More robust NaN/Inf handling
	debug = getattr(globals(), '_debug_attention', False)
	
	if not torch.isfinite(x).all():
		if debug:
			print(f"safe_softmax: Handling non-finite values. NaN: {torch.isnan(x).sum()}, Inf: {torch.isinf(x).sum()}")
		# Replace NaN with very negative values, clip extreme values
		x = torch.nan_to_num(x, nan=-1e9, posinf=1e3, neginf=-1e3)
	
	# Stabilize by subtracting max
	x_max = torch.amax(x, dim=dim, keepdim=True)
	if not torch.isfinite(x_max).all():
		if debug:
			print(f"safe_softmax: Non-finite max values detected")
		x_max = torch.nan_to_num(x_max, nan=0.0, posinf=1e3, neginf=-1e3)
	
	x_stable = x - x_max
	result = torch.nn.functional.softmax(x_stable, dim=dim)
	
	# Final check
	if not torch.isfinite(result).all():
		if debug:
			print(f"safe_softmax: Output still has non-finite values!")
		result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
		# Renormalize to ensure probabilities sum to 1
		result = result / (result.sum(dim=dim, keepdim=True) + 1e-8)
	return result
        
def replace_attention_processors(module, processor, store_attention_weights=True, **kwargs):
	attn_processors = module.attn_processors
	for k, v in attn_processors.items():
		if "attn1" in k:
			# Create processor with layer name
			attn_processors[k] = processor(store_attention_weights, layer_name=k, **kwargs)
	module.set_attn_processor(attn_processors)


def visualize_all_attn1_layers(model, save_dir="attention_visualizations", head_idx=0):
	"""
	Visualize attention patterns for all attn1 layers in the model.
	
	Args:
		model: The model with SamplewiseAttnProcessor2_0 processors
		save_dir: Directory to save visualizations
		head_idx: Which attention head to visualize
	"""
	import os
	import matplotlib.pyplot as plt
	import seaborn as sns
	
	os.makedirs(save_dir, exist_ok=True)
	
	# Get all attn1 processors
	attn_processors = model.attn_processors
	attn1_processors = {k: v for k, v in attn_processors.items() if "attn1" in k and hasattr(v, 'weights')}
	
	if not attn1_processors:
		print("No attn1 processors with stored weights found")
		return
	
	print(f"Found {len(attn1_processors)} attn1 layers to visualize")
	
	# Create a grid layout for all visualizations
	num_layers = len(attn1_processors)
	cols = min(4, num_layers)  # Max 4 columns
	rows = (num_layers + cols - 1) // cols
	
	fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
	if num_layers == 1:
		axes = [axes]
	else:
		axes = axes.flatten()
	
	layer_stats = {}
	
	for idx, (layer_name, processor) in enumerate(attn1_processors.items()):
		if processor.weights is None:
			print(f"No weights stored for {layer_name}")
			continue
			
		# Get cross-sample attention
		sample_attn = processor.get_cross_sample_attention(head_idx)
		if sample_attn is None:
			continue
			
		# Individual layer visualization
		ax = axes[idx] if idx < len(axes) else None
		if ax is not None:
			sns.heatmap(sample_attn.numpy(), 
						annot=True, 
						fmt='.3f',
						cmap='Blues',
						ax=ax,
						xticklabels=[f'S{i}' for i in range(processor.batch_size)],
						yticklabels=[f'S{i}' for i in range(processor.batch_size)],
						cbar=True)
			
			# Clean up layer name for title
			clean_name = layer_name.replace('.processor', '').replace('.', '_')
			ax.set_title(f'{clean_name}', fontsize=10, fontweight='bold')
			ax.set_xlabel('Attended Sample')
			ax.set_ylabel('Attending Sample')
		
		# Save individual layer visualization
		plt.figure(figsize=(8, 6))
		sns.heatmap(sample_attn.numpy(), 
					annot=True, 
					fmt='.3f',
					cmap='Blues',
					xticklabels=[f'Sample {i}' for i in range(processor.batch_size)],
					yticklabels=[f'Sample {i}' for i in range(processor.batch_size)])
		
		clean_name = layer_name.replace('.processor', '').replace('.', '_')
		plt.title(f'Cross-Sample Attention - {clean_name} (Head {head_idx})')
		plt.xlabel('Attended Sample')
		plt.ylabel('Attending Sample')
		
		save_path = os.path.join(save_dir, f"{clean_name}_head{head_idx}.png")
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		# Collect statistics with NaN protection
		self_attn = torch.diag(sample_attn).mean().item()
		mask = torch.eye(8).bool()
		cross_attn = sample_attn[~mask].mean().item()
		
		# Protect against NaN/Inf in statistics
		if not (torch.isfinite(torch.tensor(self_attn)) and torch.isfinite(torch.tensor(cross_attn))):
			print(f"WARNING: Non-finite values in statistics for {clean_name}")
			self_attn = 0.0 if not torch.isfinite(torch.tensor(self_attn)) else self_attn
			cross_attn = 0.0 if not torch.isfinite(torch.tensor(cross_attn)) else cross_attn
		
		# Safe ratio calculation
		if abs(cross_attn) < 1e-12:  # Effectively zero
			ratio = float('inf') if self_attn > 0 else 0.0
		else:
			ratio = self_attn / cross_attn
			if not torch.isfinite(torch.tensor(ratio)):
				ratio = 0.0
		
		layer_stats[clean_name] = {
			'self_attention': self_attn,
			'cross_attention': cross_attn,
			'self_cross_ratio': ratio
		}
		
		print(f"Saved: {save_path}")
	
	# Hide empty subplots
	for idx in range(len(attn1_processors), len(axes)):
		axes[idx].set_visible(False)
	
	# Save combined visualization
	plt.figure(fig.number)
	plt.tight_layout()
	combined_path = os.path.join(save_dir, f"all_attn1_layers_head{head_idx}.png")
	plt.savefig(combined_path, dpi=300, bbox_inches='tight')
	plt.show()
	
	# Print statistics summary
	print("\nAttention Statistics Summary:")
	print("-" * 80)
	print(f"{'Layer':<30} {'Self Attn':<12} {'Cross Attn':<12} {'Ratio':<8}")
	print("-" * 80)
	for layer, stats in layer_stats.items():
		print(f"{layer:<30} {stats['self_attention']:<12.3f} {stats['cross_attention']:<12.3f} {stats['self_cross_ratio']:<8.2f}")
	
	return layer_stats


def get_layer_attention_processors(model):
	"""
	Get all attention processors organized by layer type.
	"""
	attn_processors = model.attn_processors
	
	organized = {
		'down_blocks': {},
		'mid_block': {},
		'up_blocks': {}
	}
	
	for name, processor in attn_processors.items():
		if hasattr(processor, 'weights'):
			if 'down_blocks' in name:
				organized['down_blocks'][name] = processor
			elif 'mid_block' in name:
				organized['mid_block'][name] = processor
			elif 'up_blocks' in name:
				organized['up_blocks'][name] = processor
	
	return organized


def diagnose_attention_nan(model, verbose=True):
	"""
	Diagnose potential sources of NaN values in attention processors.
	"""
	print("=== Attention NaN Diagnosis ===")
	
	attn_processors = model.attn_processors
	attn1_processors = {k: v for k, v in attn_processors.items() if "attn1" in k and hasattr(v, 'weights')}
	
	issues_found = []
	
	for layer_name, processor in attn1_processors.items():
		layer_issues = []
		
		if processor.weights is None:
			layer_issues.append("No weights stored")
			continue
		
		# Check raw attention weights
		weights = processor.weights
		if not torch.isfinite(weights).all():
			nan_count = torch.isnan(weights).sum().item()
			inf_count = torch.isinf(weights).sum().item()
			layer_issues.append(f"Non-finite weights: {nan_count} NaN, {inf_count} Inf")
		
		# Check dimensions consistency
		if hasattr(processor, 'batch_size') and hasattr(processor, 'sequence_length'):
			expected_total_len = processor.batch_size * processor.sequence_length
			actual_len = weights.shape[2]  # [1, heads, seq_len, seq_len]
			if expected_total_len != actual_len:
				layer_issues.append(f"Dimension mismatch: expected {expected_total_len}, got {actual_len}")
		
		# Check if sequence_length is valid
		if hasattr(processor, 'sequence_length'):
			if processor.sequence_length is None or processor.sequence_length <= 0:
				layer_issues.append(f"Invalid sequence_length: {processor.sequence_length}")
		
		# Try to compute cross-sample attention and check for issues
		try:
			sample_attn = processor.get_cross_sample_attention(head_idx=0)
			if sample_attn is not None:
				if not torch.isfinite(sample_attn).all():
					nan_count = torch.isnan(sample_attn).sum().item()
					inf_count = torch.isinf(sample_attn).sum().item()
					layer_issues.append(f"Non-finite cross-sample attention: {nan_count} NaN, {inf_count} Inf")
		except Exception as e:
			layer_issues.append(f"Error computing cross-sample attention: {str(e)}")
		
		if layer_issues:
			issues_found.append((layer_name, layer_issues))
			if verbose:
				clean_name = layer_name.replace('.processor', '').replace('.', '_')
				print(f"\n{clean_name}:")
				for issue in layer_issues:
					print(f"  - {issue}")
	
	if not issues_found:
		print("✓ No obvious NaN-related issues found in attention processors")
	else:
		print(f"\n⚠ Found issues in {len(issues_found)} layers")
	
	return issues_found


def reset_attention_debug():
	"""
	Enable/disable debugging in safe_softmax and attention computation.
	"""
	global _debug_attention
	_debug_attention = not getattr(globals(), '_debug_attention', False)
	print(f"Attention debugging: {'ON' if _debug_attention else 'OFF'}")


# Global debug flag
_debug_attention = False


class SamplewiseAttnProcessor2_0:
	r"""
	    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
	    """

	def __init__(self, store_attn_weights=False, layer_name="unknown"):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
		self.store_attn_weights = store_attn_weights
		self.weights = None
		self.batch_size = None
		self.sequence_length = None
		self.num_heads = None
		self.layer_name = layer_name

	def __call__(
			self,
			attn: Attention,
			hidden_states: torch.Tensor,
			encoder_hidden_states=None,
			attention_mask=None,
			temb=None,
	) -> torch.Tensor:

		residual = hidden_states
		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)

		input_ndim = hidden_states.ndim

		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
		else:
			batch_size = hidden_states.shape[0]

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(1, -1, attn.heads, head_dim).transpose(1, 2)
		key = key.view(1, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(1, -1, attn.heads, head_dim).transpose(1, 2)
		if attn.norm_q is not None:
			query = attn.norm_q(query)
		if attn.norm_k is not None:
			key = attn.norm_k(key)

		# Store additional info for visualization
		self.batch_size = batch_size
		self.sequence_length = query.shape[2]  # tokens per sample
		self.num_heads = attn.heads

		# Compute full attention across the entire batch
		out = F.scaled_dot_product_attention(
			query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
		)

		out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

		# Output projection
		out = attn.to_out[0](out)
		out = attn.to_out[1](out)

		if input_ndim == 4:
			out = out.transpose(-1, -2).reshape(batch_size, channel, height, width)

		if attn.residual_connection:
			out = out + residual

		out = out / attn.rescale_output_factor
		return out

class AttentionGraph:
    def __init__(self, num_nodes=32):
        self.adjacency = torch.zeros((num_nodes, num_nodes))
    
    def add_edge(self, e):
        # i, j means: i attends to j
        self.adjacency[e[0], e[1]] = 1.0
    
    def add_edges(self, es):
        for e in es:
            self.add_edge(e)

 
class CustomAttnProcessor2_0:
	"""
	Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
	"""

	def __init__(self, attention_graph=AttentionGraph(), custom_attention_mask=None, ref_attention_mask=None, ref_weight=0):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
		self.attention_graph = attention_graph
		self.ref_weight = ref_weight
		self.custom_attention_mask = custom_attention_mask
		self.ref_attention_mask = ref_attention_mask

	def __call__(
		self,
		attn: Attention,
		hidden_states,
		encoder_hidden_states=None,
		attention_mask=None,
		temb=None,
	):

        
		residual = hidden_states

		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)

		input_ndim = hidden_states.ndim


		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

		batch_size, sequence_length, channels = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)

		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = torch.clone(hidden_states)
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


		'''
			reshape encoder hidden state to a single batch
		'''
		encoder_hidden_states = encoder_hidden_states.reshape(1, -1, channels)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

  


		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		'''
			each time select 1 sample from q and compute with concated kv
			concat result hidden states afterwards
		'''
		hidden_state_list = []

		for b_idx in range(batch_size):

			query_b = query[b_idx:b_idx+1]

			if self.ref_weight > 0:
				key_ref = key.clone()
				value_ref = value.clone()

				keys = [key_ref[view_idx] for view_idx in self.ref_attention_mask]
				values = [value_ref[view_idx] for view_idx in self.ref_attention_mask]

				key_ref = torch.stack(keys)
				key_ref = key_ref.view(key_ref.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

				value_ref = torch.stack(values)
				value_ref = value_ref.view(value_ref.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

			# key_a = key.clone()
			# value_a = value.clone()
			mask = self.attention_graph.adjacency[b_idx]
			key_a = key.view(key.shape[0], batch_size, -1, key.shape[-1])[:, mask>0, :, :].view(key.shape[0], -1, key.shape[-1]).clone()
			value_a = value.view(value.shape[0], batch_size, -1, value. shape[-1])[:, mask>0, :, :].view(value.shape[0], -1, value.shape[-1]).clone()

			# key_a = key_a[max(0,b_idx-1):min(b_idx+1,batch_size)+1]

			# keys = (key_a[b_idx-1], key_a[b_idx], key_a[(b_idx+1)%batch_size])
			# values = (value_a[b_idx-1], value_a[b_idx], value_a[(b_idx+1)%batch_size])

			# if b_idx not in [0, batch_size-1, batch_size//2]:
			# 	keys = keys + (key_a[min(batch_size-2, 2*(batch_size//2) - b_idx)],)
			# 	values = values + (value_a[min(batch_size-2, 2*(batch_size//2) - b_idx)],)

			key_a = key_a.view(key_a.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

			# value_a = value_a[max(0,b_idx-1):min(b_idx+1,batch_size)+1]

			value_a = value_a.view(value_a.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]
			hidden_state_a = F.scaled_dot_product_attention(
				query_b, key_a, value_a, attn_mask=None, dropout_p=0.0, is_causal=False
			)

			if self.ref_weight > 0:
				hidden_state_ref = F.scaled_dot_product_attention(
					query_b, key_ref, value_ref, attn_mask=None, dropout_p=0.0, is_causal=False
				)

				hidden_state = (hidden_state_a + self.ref_weight * hidden_state_ref) / (1+self.ref_weight)
			else:
				hidden_state = hidden_state_a

			# the output of sdp = (batch, num_heads, seq_len, head_dim)
			# TODO: add support for attn.scale when we move to Torch 2.1

			hidden_state_list.append(hidden_state)

		hidden_states = torch.cat(hidden_state_list)


		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		hidden_states = hidden_states.to(query.dtype)

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

		if attn.residual_connection:
			hidden_states = hidden_states + residual

		hidden_states = hidden_states / attn.rescale_output_factor

		return hidden_states