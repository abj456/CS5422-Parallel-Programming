#!/bin/bash
IMPLS=("Pytorch" "Flash2")

# 變動參數: BATCH_SIZE
BATCH_SIZE=32
SEQ_LENS=512
NUM_HEADS=(16 32 64)
EMB_DIM=2048
DIR_NAME="num_heads"

# 迴圈來變化 BATCH_SIZE
for NUM_HEAD in "${NUM_HEADS[@]}"
do
    for IMPL in "${IMPLS[@]}"
    do
        # 根據 IMPL 設定不同的輸出文件名
        if [ "$IMPL" == "Pytorch" ]; then
            OUTPUT_FILE="${DIR_NAME}/benchmark_result_nh${NUM_HEAD}_pth.json"
        else
            OUTPUT_FILE="${DIR_NAME}/benchmark_result_nh${NUM_HEAD}_fl2.json"
        fi

        # 執行 python 指令
        python lab5.py \
            --batch_size ${BATCH_SIZE} \
			--seq_len ${SEQ_LENS} \
			--num_heads ${NUM_HEAD} \
			--emb_dim ${EMB_DIM} \
            --impl ${IMPL} \
            --causal \
            --repeats 30 \
            --output ${OUTPUT_FILE}
    done
done
