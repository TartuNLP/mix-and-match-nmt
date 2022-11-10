# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field, fields
from typing import List, Optional

from omegaconf import II

from fairseq.models.transformer import TransformerConfig
from fairseq.models.transformer.transformer_config import EncDecBaseConfig, DecoderConfig, QuantNoiseConfig
from fairseq.utils import safe_getattr, safe_hasattr

@dataclass
class MixAndMatchEncoderConfig(EncDecBaseConfig):
    input_output_adapter_layernorm: bool = field(default=False)
    input_output_adapter_type: Optional[str] = field(default="linear")

    layer_output_dim: Optional[int] = field(default=None)
    layer_output_dims: Optional[List[int]] = field(default=None)

    layer_input_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Used to pass layer_input_dims to individual layers"},
    )
    layer_input_dims: Optional[List[int]] = field(default=None)

    layer_ffn_embed_dims: Optional[List[int]] = field(default=None)
    layer_embed_dims: Optional[List[int]] = field(default=None)
    layer_attention_heads: Optional[List[int]] = field(default=None)

    extra_output_layernorm_layers: Optional[List[int]] = field(default=None)
    extra_output_layernorm: bool = field(default=False)

    output_dim: int = field(
        default=II("model.encoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )

    freeze: bool = field(default=False)
    freeze_embeddings: bool = field(default=False)
    freeze_layers: Optional[List[int]] = field(default=None)

    def __post_init__(self):
        if self.output_dim == II("model.encoder.embed_dim"):
            self.output_dim = self.embed_dim

@dataclass
class MixAndMatchDecoderConfig(DecoderConfig):
    freeze: bool = field(default=False)
    freeze_embeddings: bool = field(default=False)
    freeze_layers: Optional[List[int]] = field(default=None)


@dataclass
class MixAndMatchTransformerConfig(TransformerConfig):
    encoder: MixAndMatchEncoderConfig = MixAndMatchEncoderConfig()
    decoder: MixAndMatchDecoderConfig = MixAndMatchDecoderConfig()
    nonstrict_model_load: bool = field(default=False)

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = MixAndMatchDecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, MixAndMatchDecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = MixAndMatchEncoderConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, MixAndMatchEncoderConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)
            return config
        else:
            return args
