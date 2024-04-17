from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
import numpy as np
import torch
from concrete.fhe.compilation import Circuit, Configuration
from qgpt2_class import QGPT2, QGPT2Seq
from quant_framework import DualArray
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.pytorch_utils import Conv1D
from utility_functions import slice_ordered_dict

import brevitas
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class QGPT2Attention(GPT2Attention):
    """Base class for building a torch module for the quantized attention mechanism."""

    def __init__(self, config: GPT2Config):
        """Initialize the base class.

        Args:
            config (GPT2Config): GPT-2's configuration.
        """
        super().__init__(config)

        self.fhe = "disable"
        self.true_float = False

    def set_fhe_mode(self, fhe: str = "disable", true_float: bool = False):
        """Set the FHE mode for the module's forward pass.

        fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
            to "disable".
        true_float (bool): If the FHE mode is set to "disable", indicate if the operations
            should be in floating points instead of being quantized. Default to False.
        """
        assert fhe in [
            "disable",
            "simulate",
            "execute",
        ], "Parameter 'fhe' can only be 'disable', 'simulate' or 'execute'."

        self.fhe = fhe
        self.true_float = true_float

class MultiHeadsAttention(QGPT2):
    """Class representing the multi-head mechanism implemented with quantization methods."""

    def run_numpy(self, q_inputs: np.ndarray) -> Union[np.ndarray, DualArray]:
        """Run the quantized operators that will be converted to FHE.

        This method is essentially the multi-head attention mechanism initially implemented in the
        forward pass of Hugging Face's GPT2Attention module but with quantized operators only

        Args:
            q_inputs (np.ndarray): The quantized inputs.

        Returns:
            Union[np.ndarray, DualArray]: The quantized outputs.
        """

        # Convert the inputs to a DualArray instance using the stored calibration data
        # q_x has shape (n_batch, n_seq, n_embed)
        q_x = DualArray(float_array=self.x_calib, int_array=q_inputs, quantizer=self.quantizer)

        # Extract the attention base module name
        mha_module_name = f"transformer.h.{self.layer}.attn."

        # Apply the first projection in order to extract Q, K and V as a single array
        # q_qkv has shape (n_batch, n_seq, 3*n_embed)
        q_qkv = q_x.linear(
            weight=self.q_weights[mha_module_name + "c_attn.weight"],
            bias=self.q_weights[mha_module_name + "c_attn.bias"],
            key=f"attention_qkv_proj_layer_{self.layer}",
        )

        # Extract Q, K and V, with shapes (n_batch, n_seq, n_embed)
        q_q, q_k, q_v = q_qkv.enc_split(3, axis=-1, key=f"qkv_split_layer_{self.layer}")

        # Reshape Q, K and V in order to be splitted into 12 heads, as done in the initial
        # implementation
        # q_q_mh, q_k_mh, q_v_mh have shape (n_batch, n_head, n_seq, n_embed // n_head)
        splitted_head_shape = (
            q_x.shape[0],
            q_x.shape[1],
            self.config.n_head,
            q_x.shape[2] // self.config.n_head,
        )
        q_q_mh = q_q.reshape(
            splitted_head_shape, key=f"q_head_reshape_layer_{self.layer}"
        ).transpose((0, 2, 1, 3), key=f"q_head_transpose_layer_{self.layer}")
        q_k_mh = q_k.reshape(
            splitted_head_shape, key=f"k_head_reshape_layer_{self.layer}"
        ).transpose((0, 2, 1, 3), key=f"k_head_transpose_layer_{self.layer}")
        q_v_mh = q_v.reshape(
            splitted_head_shape, key=f"v_head_reshape_layer_{self.layer}"
        ).transpose((0, 2, 1, 3), key=f"v_head_transpose_layer_{self.layer}")

        # Compute the attention
        # q_y_mh has shape (n_batch, n_head, n_seq, n_embed // n_head)
        q_y_mh = self.attention(q_q_mh, q_k_mh, q_v_mh)

        # Merge back the 12 heads along axis -1, as done in the initial implementation
        # q_y has shape (n_batch, n_seq, n_embed)
        q_y_mh = q_y_mh.transpose((0, 2, 1, 3), key=f"head_transpose_layer_{self.layer}")
        q_y = q_y_mh.reshape(q_x.shape, key=f"head_reshape_layer_{self.layer}")

        # Re-quantize the result to n_bits for precision stability
        q_y = q_y.requant(key="q_y_requant")

        # Apply the last projection
        # q_y has shape (n_batch, n_seq, n_embed)
        q_y = q_y.linear(
            weight=self.q_weights[mha_module_name + "c_proj.weight"],
            bias=self.q_weights[mha_module_name + "c_proj.bias"],
            key=f"attention_last_proj_layer_{self.layer}",
        )

        return self.finalize(q_y)

class MultiHeadsAttentionSeq(QGPT2Seq):
    """Class representing the multi-head mechanism implemented with quantization methods."""

    def run_numpy(self, q_inputs: np.ndarray) -> Union[np.ndarray, DualArray]:
        """Run the quantized operators that will be converted to FHE.

        This method is essentially the multi-head attention mechanism initially implemented in the
        forward pass of Hugging Face's GPT2Attention module but with quantized operators only

        Args:
            q_inputs (np.ndarray): The quantized inputs.

        Returns:
            Union[np.ndarray, DualArray]: The quantized outputs.
        """

        # Convert the inputs to a DualArray instance using the stored calibration data
        # q_x has shape (n_batch, n_seq, n_embed)
        q_x = DualArray(float_array=self.x_calib, int_array=q_inputs, quantizer=self.quantizer)

        # Extract the attention base module name
        mha_module_name = f"transformer.h.{self.layer}.attn."

        # Apply the first projection in order to extract Q, K and V as a single array
        # q_qkv has shape (n_batch, n_seq, 3*n_embed)
        q_qkv = q_x.linear(
            weight=self.q_weights[mha_module_name + "c_attn.weight"],
            bias=self.q_weights[mha_module_name + "c_attn.bias"],
            key=f"attention_qkv_proj_layer_{self.layer}",
        )

        # Extract Q, K and V, with shapes (n_batch, n_seq, n_embed)
        q_q, q_k, q_v = q_qkv.enc_split(3, axis=-1, key=f"qkv_split_layer_{self.layer}")

        # Reshape Q, K and V in order to be splitted into 12 heads, as done in the initial
        # implementation
        # q_q_mh, q_k_mh, q_v_mh have shape (n_batch, n_head, n_seq, n_embed // n_head)
        splitted_head_shape = (
            q_x.shape[0],
            q_x.shape[1],
            self.config.n_head,
            q_x.shape[2] // self.config.n_head,
        )
        q_q_mh = q_q.reshape(
            splitted_head_shape, key=f"q_head_reshape_layer_{self.layer}"
        ).transpose((0, 2, 1, 3), key=f"q_head_transpose_layer_{self.layer}")
        q_k_mh = q_k.reshape(
            splitted_head_shape, key=f"k_head_reshape_layer_{self.layer}"
        ).transpose((0, 2, 1, 3), key=f"k_head_transpose_layer_{self.layer}")
        q_v_mh = q_v.reshape(
            splitted_head_shape, key=f"v_head_reshape_layer_{self.layer}"
        ).transpose((0, 2, 1, 3), key=f"v_head_transpose_layer_{self.layer}")

        # Compute the attention
        # q_y_mh has shape (n_batch, n_head, n_seq, n_embed // n_head)
        q_y_mh = self.attention(q_q_mh, q_k_mh, q_v_mh)

        # Merge back the 12 heads along axis -1, as done in the initial implementation
        # q_y has shape (n_batch, n_seq, n_embed)
        q_y_mh = q_y_mh.transpose((0, 2, 1, 3), key=f"head_transpose_layer_{self.layer}")
        q_y = q_y_mh.reshape(q_x.shape, key=f"head_reshape_layer_{self.layer}")

        # Re-quantize the result to n_bits for precision stability
        q_y = q_y.requant(key="q_y_requant")

        # Apply the last projection
        # q_y has shape (n_batch, n_seq, n_embed)
        q_y = q_y.linear(
            weight=self.q_weights[mha_module_name + "c_proj.weight"],
            bias=self.q_weights[mha_module_name + "c_proj.bias"],
            key=f"attention_last_proj_layer_{self.layer}",
        )

        return self.finalize(q_y)

class QGPT2MultiHeadsAttention(QGPT2Attention):
    """Torch module that rewrites GPT-2's multi-head attention with quantized operations."""

    def __init__(self, config, layer, n_bits=16):
        super().__init__(config)

        # Instantiate the quantized module used for the multi-head attention mechanism
        self.q_module = MultiHeadsAttention(n_bits=n_bits, layer=layer)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """GPT-2's multi-head attention pass made for FHE computations.

        The initial implementation can be found in huggingFace's GPT2Attention class.
        """
        if encoder_hidden_states is not None:
            raise ValueError(
                "Class cannot be used as cross attention, please make sure to not instantiate "
                "class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        if self.reorder_and_upcast_attn:
            raise ValueError("Method 'reorder_and_upcast_attn' is not implemented")

        # Apply the multi-head attention using FHE-compliant operators
        attn_output = self.q_module.run_torch(
            hidden_states,
            fhe=self.fhe,
            true_float=self.true_float,
        )

        # The method does not handle the cache option
        return (attn_output, None)

class QGPT2MultiHeadsAttentionSeq(QGPT2Attention):
    """Torch module that rewrites GPT-2's multi-head attention with quantized operations."""

    def __init__(self, config, layer, n_bits=16):
        super().__init__(config)

        # Instantiate the quantized module used for the multi-head attention mechanism
        self.q_module = MultiHeadsAttentionSeq(n_bits=n_bits, layer=layer)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """GPT-2's multi-head attention pass made for FHE computations.

        The initial implementation can be found in huggingFace's GPT2Attention class.
        """
        if encoder_hidden_states is not None:
            raise ValueError(
                "Class cannot be used as cross attention, please make sure to not instantiate "
                "class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        if self.reorder_and_upcast_attn:
            raise ValueError("Method 'reorder_and_upcast_attn' is not implemented")

        # Apply the multi-head attention using FHE-compliant operators
        attn_output = self.q_module.run_torch(
            hidden_states,
            fhe=self.fhe,
            true_float=self.true_float,
        )

        # The method does not handle the cache option
        return (attn_output, None)

class QGPT2LMHeadModel(GPT2LMHeadModel):
    """Base class for integrating quantized operations within GPT2LMHeadModel's forward pass."""

    def __init__(
        self,
        config: GPT2Config,
        n_bits: int,
        layers: int = 12
    ):
        """Initialize the base class.

        This class essentially overwrites GPT-2's attention module found in the layer whose index is
        given with the given quantized module.

        Args:
            config (GPT2Config): GPT-2's configuration.
            n_bits (int): The number of bits to use for quantizing the inputs, weights and
                activations.
            attention (Union[QGPT2SingleHeadAttention, QGPT2MultiHeadsAttention]): The quantized attention module
                to consider.
            layer (int): The index representing the GPT-2 layer to consider. Default to 0.
        """
        assert 0 <= layers <= config.n_layer

        super().__init__(config)
        self.config = config
        self.quantized_layers = layers

        for i in range(layers):
            curr_layer = self.quantized_layers - 1 - i
            self.transformer.h[curr_layer].attn = QGPT2MultiHeadsAttention(config, n_bits=n_bits, layer=curr_layer)
            
    def set_fhe_mode(self, fhe: str = "disable", true_float: bool = False):
        """Set the FHE mode for the module's forward pass.

        fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
            to "disable".
        true_float (bool): If the FHE mode is set to "disable", indicate if the operations
            should be in floating points instead of being quantized. Default to False.
        """
        for i in range(self.quantized_layers):
            curr_layer = self.quantized_layers - 1 - i
            self.transformer.h[curr_layer].attn.set_fhe_mode(fhe=fhe, true_float=true_float)

    def compile(
        self, inputset_ids: torch.Tensor, configuration: Optional[Configuration] = None
    ) -> Circuit:
        """Compile the model using the stored calibration data.

        Args:
            inputset_ids (torch.Tensor): The token ids to consider as an inputset.
            configuration (Optional[Configuration]): The configuration to use during compilation.
                Default to None.

        Returns:
            Circuit: The underlying FHE circuit.
        """

        # Disable the FHE execution, as the following forward pass should be made in the clear along
        # floating point values. This is done in order to properly calibrate and store the
        # quantization parameters such as the scale and zero points
        self.set_fhe_mode(fhe="disable", true_float=False)

        # Execute a full pass in the clear
        self.forward(inputset_ids, use_cache=False)

        circuits = []

        # Compile the attention module using stored calibration data (made of intermediary hidden
        # states)
        for i in tqdm(range(self.quantized_layers)):
            curr_layer = self.quantized_layers - 1 - i
            circuits.append(self.transformer.h[curr_layer].attn.q_module.compile(configuration=configuration))

        return circuits

class QGPT2MultiHeadsAttentionSeq(QGPT2Attention):
    """Torch module that rewrites GPT-2's multi-head attention with quantized operations."""

    def __init__(self, config, layer, n_bits=16):
        super().__init__(config)

        # Instantiate the quantized module used for the multi-head attention mechanism
        self.q_module = MultiHeadsAttentionSeq(n_bits=n_bits, layer=layer)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """GPT-2's multi-head attention pass made for FHE computations.

        The initial implementation can be found in huggingFace's GPT2Attention class.
        """
        if encoder_hidden_states is not None:
            raise ValueError(
                "Class cannot be used as cross attention, please make sure to not instantiate "
                "class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        if self.reorder_and_upcast_attn:
            raise ValueError("Method 'reorder_and_upcast_attn' is not implemented")

        # Apply the multi-head attention using FHE-compliant operators
        attn_output = self.q_module.run_torch(
            hidden_states,
            fhe=self.fhe,
            true_float=self.true_float,
        )

        # The method does not handle the cache option
        return (attn_output, None)

class QGPT2ForSequenceClassification(GPT2ForSequenceClassification):
    """Base class for integrating quantized operations within GPT2LMHeadModel's forward pass."""

    def __init__(
        self,
        config: GPT2Config,
        n_bits: int,
        layers: List[int] = range(12),
        quantize_head: bool = True
    ):
        """Initialize the base class.

        This class essentially overwrites GPT-2's attention module found in the layer whose index is
        given with the given quantized module.

        Args:
            config (GPT2Config): GPT-2's configuration.
            n_bits (int): The number of bits to use for quantizing the inputs, weights and
                activations.
            attention (Union[QGPT2SingleHeadAttention, QGPT2MultiHeadsAttention]): The quantized attention module
                to consider.
            layer (int): The index representing the GPT-2 layer to consider. Default to 0.
        """
        for layer in layers:
            assert 0 <= layer <= config.n_layer

        super().__init__(config)
        self.config = config
        self.quantized_layers = layers

        for i in layers:
            self.transformer.h[i].attn = QGPT2MultiHeadsAttentionSeq(config, n_bits=n_bits, layer=i)
            
    def set_fhe_mode(self, fhe: str = "disable", true_float: bool = False):
        """Set the FHE mode for the module's forward pass.

        fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
            to "disable".
        true_float (bool): If the FHE mode is set to "disable", indicate if the operations
            should be in floating points instead of being quantized. Default to False.
        """
        for i in self.quantized_layers:
            self.transformer.h[i].attn.set_fhe_mode(fhe=fhe, true_float=true_float)

    def compile(
        self, inputset_ids: torch.Tensor, configuration: Optional[Configuration] = None
    ) -> Circuit:
        """Compile the model using the stored calibration data.

        Args:
            inputset_ids (torch.Tensor): The token ids to consider as an inputset.
            configuration (Optional[Configuration]): The configuration to use during compilation.
                Default to None.

        Returns:
            Circuit: The underlying FHE circuit.
        """

        # Disable the FHE execution, as the following forward pass should be made in the clear along
        # floating point values. This is done in order to properly calibrate and store the
        # quantization parameters such as the scale and zero points
        self.set_fhe_mode(fhe="disable", true_float=False)

        # Execute a full pass in the clear
        self.forward(inputset_ids, use_cache=False)

        circuits = []

        # Compile the attention module using stored calibration data (made of intermediary hidden
        # states)
        for i in tqdm(self.quantized_layers):
            circuits.append(self.transformer.h[i].attn.q_module.compile(configuration=configuration))

        return circuits
    