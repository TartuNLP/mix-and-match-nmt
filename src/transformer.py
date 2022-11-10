# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from fairseq.models.lstm import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from .transformer_config import MixAndMatchTransformerConfig
from .transformer_decoder import MixAndMatchTransformerDecoderBase
from .transformer_encoder import MixAndMatchTransformerEncoderBase
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModelBase, DEFAULT_MIN_PARAMS_TO_WRAP, transformer_legacy,
)


def freeze_module(module):
    if not hasattr(module, "named_parameters"):
        return
    for _, param in module.named_parameters():
        param.requires_grad = False


class MixAndMatchTransformerModelBase(TransformerModelBase):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, MixAndMatchTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        cfg.encoder.output_dim = int(cfg.encoder.output_dim)
        mdl = super().build_model(cfg, task)
        encoder, decoder = mdl.encoder, mdl.decoder

        if cfg.encoder.freeze:
            freeze_module(encoder)
        if cfg.decoder.freeze:
            freeze_module(decoder)
        if cfg.encoder.freeze_layers is not None:
            for layer_idx in cfg.encoder.freeze_layers:
                freeze_module(encoder.layers[layer_idx])
        if cfg.decoder.freeze_layers is not None:
            for layer_idx in cfg.decoder.freeze_layers:
                freeze_module(decoder.layers[layer_idx])
        if cfg.encoder.freeze_embeddings:
            freeze_module(encoder.embed_tokens)
            freeze_module(encoder.embed_positions)
            freeze_module(encoder.layernorm_embedding)
        if cfg.decoder.freeze_embeddings:
            freeze_module(decoder.embed_tokens)
            freeze_module(decoder.embed_positions)
            freeze_module(decoder.layernorm_embedding)
            freeze_module(decoder.output_projection)

        return mdl

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return MixAndMatchTransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return MixAndMatchTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    def load_state_dict(
            self,
            state_dict,
            strict=True,
            model_cfg=None,
            args=None,
    ):
        super().load_state_dict(
            state_dict,
            strict and not self.cfg.nonstrict_model_load,
            model_cfg,
            args
        )


@register_model("mix_and_match_transformer")
class MixAndMatchTransformerModel(MixAndMatchTransformerModelBase):
    @staticmethod
    def cfg_from_namespace(args):
        return MixAndMatchTransformerConfig.from_namespace(args)

    def __init__(self, args, encoder, decoder):
        cfg = self.cfg_from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        return super().build_model(cls.cfg_from_namespace(args), task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            cls.cfg_from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            cls.cfg_from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            cls.cfg_from_namespace(args), tgt_dict, embed_tokens
        )


@register_model_architecture("mix_and_match_transformer", "mix_and_match_transformer")
def base_architecture(args):
    transformer_legacy.base_architecture(args)
    for arg_name, arg_default_value in (
        ("encoder_input_output_adapter_layernorm", False),
        ("encoder_input_output_adapter_type", "linear"),
        ("encoder_layer_output_dim", None),
        ("encoder_layer_output_dims", None),
        ("encoder_layer_ffn_embed_dims", None),
        ("encoder_layer_embed_dims", None),
        ("encoder_layer_attention_heads", None),
        ("encoder_layer_attention_heads", None),
        ("encoder_freeze", False),
        ("encoder_freeze_embeddings", False),
        ("encoder_freeze_layers", None),
        ("decoder_freeze", False),
        ("decoder_freeze_embeddings", False),
        ("decoder_freeze_layers", None),
        ("encoder_extra_output_layernorm", None),
        ("encoder_extra_output_layernorm_layers", None),
        ("encoder_output_dim", args.encoder_embed_dim),
        ("nonstrict_model_load", False),
    ):
        setattr(args, arg_name, getattr(args, arg_name, arg_default_value))

