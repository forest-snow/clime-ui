#!/bin/bash
python prepare_ui.py \
    --src-emb embeds/en \
    --tgt-emb embeds/fr \
    -k 5 \
    --task example \
    --rank data/word_rank.txt \
    --max 30 \
    --src-doc data/en.json \
    --tgt-doc data/fr_unlabeled.json \
    --src-f 10000 \
    --tgt-f 10000 \
    --src-lang ENGLISH \
    --tgt-lang FRENCH \
    --categories data/categories.txt

