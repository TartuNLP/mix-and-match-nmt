# Mix and Match NMT

This repository allows for you to cobmine different pre-trained NMT models.
### Requirements
#### Fairseq
Install fairseq 0.12.1 either with pip:
```
pip install fairseq==0.12.1
```
or from the fairseq github commit corresponding to 12.0.1 (336c26a5e66634d904bac8c462cd319c62a31fb7). Other versions have not been tested.
See the official repository [fairseq](https://github.com/facebookresearch/fairseq) for further instructions.

#### Using this repository
Clone this repository and reference it as the fairseq user directory when calling fairseq by adding:
```
--user-dir ${repo_dir}/src
```
where `${repo_dir}` is the directory of this repository.

### Creating checkpoints
`scripts/create_mix_match_checkpoint.py` lets you create a checkpoint that combines encoder from one model and decoder from another.

Checkpoints used:
* [MTee-general](https://huggingface.co/tartuNLP/mtee-general)
* NLLB-200-Distilled Dense 1.3B ([repo](https://github.com/facebookresearch/fairseq/tree/nllb), [direct download](https://tinyurl.com/nllb200densedst1bcheckpoint))

For example, creating the mix-and-match model from the main experiments.
```
python -u ${repo_dir}/scripts/create_mix_match_checkpoint.py \
  --encoder-model-path ${nllb_model_dir}/nllb_1B_dense_distil.pt \
  --decoder-model-path ${mtee_model_dir}/modular_model.pt \
  --decoder-prefix models.en-et.decoder \
  --decoder-rename-prefix decoder \
  --out-path ${model_dir}/checkpoint.pt \
  --extra-rename-prefixes "{'encoder.layer_norm': 'encoder.layers.11.extra_output_layernorm'}"
```

### Training models

Training arguments:
* Model structure:
  * `--encoder-input-output-adapter-type` - adapter type, 'linear' by default (recommended)
  * `--encoder-layer-embed-dims` - comma delimited list of embed-dims for each layer
  * `--encoder-layer-ffn-embed-dims` - comma delimited list of ffn-embed-dims for each layer
  * `--encoder-layer-input-dims` - comma delimited list of input dimensions for each layer, if does not match with embed-dim, dimension adapter before that layer.
  * `--encoder-output-dim` - final output dimension of the encoder (for creating the decoder model)
   * `--disable-tgt-augmentation` `--disable-src-augmentation` -  disables adding symbols to the src or tgt directory (needed for loading some models), creates vocabulary exactly like in the given dictionary.
* freezing parameters
  * `--encoder-freeze-layers` - comma delimited list of encoder layers to freeze
  * `--encoder-freeze-embeddings` 
  * `--decoder-freeze-embeddings` 
  * `--decoder-freeze` 
  * `--encoder-freeze`
* `--nonstrict-model-load` - allows loading some parameters of the model, leaving others randomly initialised. 
We suggest running the train command without this first so you can check which parameters are loaded.

#### Example of two stage training
First stage:
```
fairseq-train ${bin_dir} \
    --task mix_and_match_multilingual_translation --arch mix_and_match_transformer \
    --user-dir ${repo_dir}/src \
    --max-update 50000 \
    --save-interval-updates 5000 \
    --validate-interval-updates 5000 \
    --keep-interval-updates 1 \
    --seed 1 \
    --finetune-from-model ${model_dir}/checkpoint.pt \
    --save-dir ${save_dir_stage_1} \
    --lang-pairs deu_Latn-est_Latn,eng_Latn-est_Latn,pol_Latn-est_Latn,fra_Latn-est_Latn \
    --wandb-project pretrained \
    --max-tokens 4096 --update-freq 1 \
    --encoder-normalize-before \
    --encoder-embed-dim 1024 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 2048 \
    --encoder-layers 28 --decoder-layers 6 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --encoder-layerdrop 0 --decoder-layerdrop 0 \
    --source-dict ${model_dir}/model_dict.src.txt \
    --target-dict ${model_dir}/model_dict.tgt.txt \
    --nonstrict-model-load \
    --disable-tgt-augmentation \
    --share-decoder-input-output-embed \
    --encoder-layer-embed-dims 1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,512,512,512,512 \
    --encoder-layer-ffn-embed-dims 8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,2048,2048,2048,2048 \
    --encoder-freeze-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --encoder-freeze-embeddings --decoder-freeze --decoder-freeze-embeddings \
    --encoder-layer-input-dims 1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,512,512,512 \
    --encoder-input-output-adapter-type linear \
    --encoder-output-dim 512 \
    --encoder-extra-output-layernorm-layers 23 \
    --lang-tok-style multilingual --encoder-langtok src \
    --attention-dropout 0.1 --activation-dropout 0.0 --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps '1e-08' --weight-decay 0.0 \
    --lr 0.0005 --warmup-updates 4000 --warmup-init-lr '1e-07' --lr-scheduler inverse_sqrt \
    --fp16 \
    --criterion cross_entropy \
    --clip-norm 1.0 \
    --ddp-backend=no_c10d --num-workers 1
```

Second stage:
```
fairseq-train ${bin_dir} \
    --task mix_and_match_multilingual_translation --arch mix_and_match_transformer \
    --user-dir ${repo_dir}/src \
    --max-update 50000 \
    --save-interval-updates 5000 \
    --validate-interval-updates 5000 \
    --keep-interval-updates 1 \
    --seed 1 \
    --finetune-from-model ${save_dir_stage_1}/checkpoint_last.pt \
    --save-dir ${save_dir_stage_2} \
    --lang-pairs deu_Latn-est_Latn,eng_Latn-est_Latn,pol_Latn-est_Latn,fra_Latn-est_Latn \
    --wandb-project pretrained \
    --max-tokens 4096 --update-freq 1 \
    --encoder-normalize-before \
    --encoder-embed-dim 1024 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 2048 \
    --encoder-layers 28 --decoder-layers 6 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --encoder-layerdrop 0 --decoder-layerdrop 0 \
    --source-dict ${model_dir}/model_dict.src.txt \
    --target-dict ${model_dir}/model_dict.tgt.txt \
    --nonstrict-model-load \
    --disable-tgt-augmentation \
    --share-decoder-input-output-embed \
    --encoder-layer-embed-dims 1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,512,512,512,512 \
    --encoder-layer-ffn-embed-dims 8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,2048,2048,2048,2048 \
    --encoder-freeze-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --encoder-freeze-embeddings \
    --encoder-layer-input-dims 1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,512,512,512 \
    --encoder-input-output-adapter-type linear \
    --encoder-output-dim 512 \
    --encoder-extra-output-layernorm-layers 23 \
    --lang-tok-style multilingual --encoder-langtok src \
    --attention-dropout 0.1 --activation-dropout 0.0 --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps '1e-08' --weight-decay 0.0 \
    --lr 0.0001 --warmup-updates 4000 --warmup-init-lr '1e-07' --lr-scheduler inverse_sqrt \
    --fp16 \
    --criterion cross_entropy \
    --clip-norm 1.0 \
    --ddp-backend=no_c10d --num-workers 1
```

Model used is combination of NLLB-1B-distilled and MTee-general (see [Creating checkpoints](#creating-checkpoints)).

Source files are processed 
with sentencepiece model of [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) ([direct link](https://tinyurl.com/flores200sacrebleuspm) ).
Target files are normalized with 
[normalization script](https://github.com/Project-MTee/translation-worker/blob/main/nmt_worker/normalization.py) 
of MTee and then pre-processed with the Estonian SentencePiece model of 
[MTee](https://huggingface.co/tartuNLP/mtee-general).

The model dictionary for target is the Estonian dictionary of MTee. 
For source side we use the dictionary of NLLB, however we append the language codes and extra tokens
in the correct order so that we don't have to rely on fairseq of adding them and not have to supply full list of languages each time.
Find it in `extra/nllb_model_dict.txt`.