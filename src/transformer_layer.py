# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

import torch.nn as nn
from torch import Tensor

from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase, TransformerEncoderLayerBase

logger = logging.getLogger(__name__)


class MixAndMatchTransformerEncoderLayerBase(TransformerEncoderLayerBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.encoder.layer_output_dim is not None and cfg.encoder.layer_output_dim != self.embed_dim:
            self.output_adapter = self.build_dim_adapter(
                self.embed_dim,
                cfg.encoder.layer_output_dim,
                self.quant_noise,
                self.quant_noise_block_size,
                adapter_type=cfg.encoder.input_output_adapter_type,
                layernorm=cfg.encoder.input_output_adapter_layernorm,
            )
        else:
            self.output_adapter = None

        if cfg.encoder.layer_input_dim is not None and cfg.encoder.layer_input_dim != self.embed_dim:
            self.input_adapter = self.build_dim_adapter(
                cfg.encoder.layer_input_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
                adapter_type=cfg.encoder.input_output_adapter_type,
                layernorm=cfg.encoder.input_output_adapter_layernorm,
            )
        else:
            self.input_adapter = None

        if cfg.encoder.extra_output_layernorm:
            self.extra_output_layernorm = LayerNorm(self.embed_dim, export=cfg.export)
        else:
            self.extra_output_layernorm = None

    def build_dim_adapter(
            self,
            input_dim,
            output_dim,
            q_noise,
            qn_block_size,
            adapter_type="linear",
            layernorm=False,
    ):
        if input_dim == output_dim:
            return None

        logger.info(
            f"Desired input/output dimensions of layer are different from actual values. "
            f"Creating adapter with input_dim={input_dim}, output_dim={output_dim}."
        )

        adapter_layers = []

        if layernorm and self.normalize_before:
            adapter_layers.append(LayerNorm(output_dim))

        if adapter_type == "linear":
            adapter_layers.extend(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim),
                        p=q_noise,
                        block_size=qn_block_size,
                    )
                ]
            )
        elif adapter_type == "MLP":
            adapter_layers.extend(
                [
                    quant_noise(
                        nn.Linear(input_dim, output_dim),
                        p=q_noise,
                        block_size=qn_block_size,
                    ),
                    nn.ReLU(),
                    quant_noise(
                        nn.Linear(output_dim, output_dim),
                        p=q_noise,
                        block_size=qn_block_size,
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown output adapter type {adapter_type}")

        if layernorm and not self.normalize_before:
            adapter_layers.append(LayerNorm(output_dim))

        return nn.Sequential(*adapter_layers)

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
    ):
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        x = super().forward(x, encoder_padding_mask, attn_mask)

        if self.extra_output_layernorm is not None:
            x = self.extra_output_layernorm(x)

        if self.output_adapter is not None:
            x = self.output_adapter(x)

        return x


class MixAndMatchTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.output_dim,
            vdim=cfg.encoder.output_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
