# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from omegaconf import OmegaConf, DictConfig

from .transformer_layer import MixAndMatchTransformerEncoderLayerBase
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerEncoderBase
from fairseq.modules import (
    LayerNorm,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper


class MixAndMatchTransformerEncoderBase(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens):
        super().__init__(cfg, dictionary, embed_tokens)

        self.layers.extend(
            [
                self.build_encoder_layer(self.get_layer_specific_cfg(cfg, i))
                for i in range(cfg.encoder.layers)
            ]
        )

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(
                cfg.encoder.output_dim, export=cfg.export
            )
        else:
            self.layer_norm = None

    def get_layer_specific_cfg(self, cfg, layer_idx):
        layer_custom_args = (
            ("layer_output_dims", "layer_output_dim"),
            ("layer_input_dims", "layer_input_dim"),
            ("layer_ffn_embed_dims", "ffn_embed_dim"),
            ("layer_embed_dims", "embed_dim"),
            ("layer_attention_heads", "attention_heads"),
        )

        layer_toggle_args = (
            ("extra_output_layernorm_layers", "extra_output_layernorm"),
        )

        def replace_cfg(cfg, **kwargs):
            if isinstance(cfg, DictConfig):
                _cfg = OmegaConf.to_container(cfg, resolve=True)
                _cfg.update(kwargs)
                return OmegaConf.create(_cfg)
            else:
                return dataclasses.replace(cfg, **kwargs)


        enc_cfg = cfg.encoder

        for arg_name, target_arg_name in layer_custom_args:
            values = getattr(cfg.encoder, arg_name)
            if values is None:
                continue
            elif isinstance(values, list):
                assert len(values) == cfg.encoder.layers
                enc_cfg = replace_cfg(
                    enc_cfg, **{target_arg_name: values[layer_idx]}
                )
            else:
                raise ValueError(f"Unsupported type for {arg_name}: {type(values)}")

        for arg_name, target_arg_name in layer_toggle_args:
            values = getattr(cfg.encoder, arg_name)
            if values is None:
                continue
            elif isinstance(values, list):
                assert all(0 <= v < cfg.encoder.layers for v in values)
                if layer_idx in values:
                    enc_cfg = replace_cfg(
                        enc_cfg, **{target_arg_name: True}
                    )
            else:
                raise ValueError(f"Unsupported type for {arg_name}: {type(values)}")

        if enc_cfg == cfg.encoder:
            return cfg

        return replace_cfg(cfg, encoder=enc_cfg)

    def build_encoder_layer(self, cfg):
        layer = MixAndMatchTransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
