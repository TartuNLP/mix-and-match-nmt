import argparse
import ast
import gc
from collections import OrderedDict
from typing import List, Optional

import torch


def eval_dict(x, key_type=str, value_type=str):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return {}
        x = ast.literal_eval(x)

    return {key_type(k): value_type(v) for k, v in x.items()}


def load_state_dict(path, keep_prefix=None, rename_prefix=None):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    state_dict = checkpoint["model"]
    if keep_prefix is None:
        return state_dict

    state_dict = OrderedDict(
        [(k, v) for k, v in state_dict.items() if k.startswith(keep_prefix)]
    )

    if rename_prefix is None:
        return state_dict

    return OrderedDict(
        [(rename_prefix + k[len(keep_prefix):], v) for k, v in state_dict.items()]
    )


def load_state_dicts(path, keep_prefixes=None, rename_prefixes=None):
    if keep_prefixes is None and rename_prefixes is None:
        keep_prefixes = [None]
        rename_prefixes = [None]

    if keep_prefixes is None:
        keep_prefixes = [None] * len(rename_prefixes)

    if rename_prefixes is None:
        rename_prefixes = [None] * len(keep_prefixes)

    if len(keep_prefixes) != len(rename_prefixes):
        raise ValueError("keep-prefixes and rename-prefixes must have the same number of elements (if not None)")

    dict_items = []
    for keep_prefix, rename_prefix in zip(keep_prefixes, rename_prefixes):
        dict_items += list(load_state_dict(path, keep_prefix, rename_prefix).items())
        gc.collect()
    return OrderedDict(dict_items)


def rename_prefix(name, prefix, rename):
    if not name.startswith(prefix):
        return name

    if len(name) == len(prefix):
        return rename

    return rename + name[len(prefix):]


def csv_str_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    return x.split(",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-model-path", default=None)
    parser.add_argument("--decoder-model-path", default=None)
    parser.add_argument("--out-path")
    parser.add_argument("--encoder-prefixes", type=csv_str_list, required=False, default="encoder")
    parser.add_argument("--decoder-prefixes", type=csv_str_list, required=False, default="decoder")
    parser.add_argument("--encoder-rename-prefixes", type=csv_str_list, required=False, default=None)
    parser.add_argument("--decoder-rename-prefixes", type=csv_str_list, required=False, default=None)
    parser.add_argument(
        "--extra-rename-prefixes", required=False, default=None, type=lambda x: eval_dict(x, str, str)
    )

    args = parser.parse_args()

    state_items = []
    if args.encoder_model_path is None and args.decoder_model_path is None:
        raise ValueError("Encoder and decoder paths can not both be None.")
    if args.encoder_model_path is not None:
        state_items += list(load_state_dicts(
            args.encoder_model_path, args.encoder_prefixes, args.encoder_rename_prefixes
        ).items())
        gc.collect()
    if args.decoder_model_path is not None:
        state_items += list((load_state_dicts(
            args.decoder_model_path, args.decoder_prefixes, args.decoder_rename_prefixes
        ).items()))
        gc.collect()
    final_state_dict = OrderedDict(state_items)

    # option for additional renames
    if args.extra_rename_prefixes is not None:
        for prefix, rename in args.extra_rename_prefixes.items():
            final_state_dict = OrderedDict(
                (rename_prefix(k, prefix, rename), v)
                for k, v in final_state_dict.items()
            )

    print("entries in final state dict:")
    for k in final_state_dict:
        print(k)

    state = torch.load(args.encoder_model_path, map_location=torch.device("cpu"))
    final_checkpoint = {
        "model": final_state_dict,
        # adding rest of the state for compatibility
        **{
            k: v for k, v in state.items() if k not in ("model", "last_optimizer_state")
        },
    }

    torch.save(final_checkpoint, args.out_path)
