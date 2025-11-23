#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}
find "$USER_CACHE_PATH" -mindepth 1 -name "sid" -prune -o -delete
pip install ujson
pip install orjson
pip install pynvml
pip install "nv_grouped_gemm-1.1.4.post4-cp310-cp310-linux_x86_64.whl"
cp $TRAIN_DATA_PATH/item_feat_dict.json $USER_CACHE_PATH/item_feat_dict.json
cp $TRAIN_DATA_PATH/seq.jsonl $USER_CACHE_PATH/seq.jsonl
cp $TRAIN_DATA_PATH/seq_offsets.pkl $USER_CACHE_PATH/seq_offsets.pkl

python item_exposure_data.py
python timestamp_buckets.py
python -u main_dist.py
# 发布第6个epoch




