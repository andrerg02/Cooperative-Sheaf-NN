#!/bin/sh

python -m exp.run \
    --dataset=tolokers \
    --task=node_level \
    --task_type=binary \
    --optimizer=adam \
    --scheduler=none \
    --batch_size=1 \
    --accum_grad=1 \
    --d=3 \
    --layers=2 \
    --gnn_layers=0 \
    --gnn_hidden=32 \
    --pe_size=0 \
    --hidden_channels=32 \
    --epochs=2000 \
    --early_stopping=200 \
    --left_weights=False \
    --right_weights=True \
    --lr=0.02 \
    --weight_decay=1e-8 \
    --input_dropout=0.2 \
    --dropout=0.2 \
    --use_act=True \
    --folds=10 \
    --model=CoopSheaf \
    --entity="${ENTITY}"