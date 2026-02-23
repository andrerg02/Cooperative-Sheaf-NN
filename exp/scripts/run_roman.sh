#!/bin/sh

python -m exp.run \
    --dataset=roman_empire \
    --task=node_level \
    --task_type=multiclass \
    --optimizer=adam \
    --scheduler=none \
    --batch_size=1 \
    --accum_grad=1 \
    --d=4 \
    --layers=5 \
    --gnn_layers=3 \
    --gnn_hidden=32 \
    --gnn_default=2 \
    --pe_size=0 \
    --hidden_channels=32 \
    --epochs=2000 \
    --early_stopping=200 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.02 \
    --weight_decay=1e-7 \
    --input_dropout=0.2 \
    --dropout=0.2 \
    --use_act=True \
    --folds=10 \
    --model=CoopSheaf \
    --stop_strategy='acc' \
    --entity="${ENTITY}"