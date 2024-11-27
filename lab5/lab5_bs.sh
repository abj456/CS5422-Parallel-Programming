#!/bin/bash
IMPLS=("Pytorch" "Flash2")

# 變動參數: BATCH_SIZE
BATCH_SIZES=(16 32 64)
SEQ_LEN=512
NUM_HEADS=32
EMB_DIM=2048
DIR_NAME="batch_size"

# 迴圈來變化 BATCH_SIZE
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    for IMPL in "${IMPLS[@]}"
    do
        # 根據 IMPL 設定不同的輸出文件名
        if [ "$IMPL" == "Pytorch" ]; then
            OUTPUT_FILE="${DIR_NAME}/benchmark_result_bs${BATCH_SIZE}_pth.json"
        else
            OUTPUT_FILE="${DIR_NAME}/benchmark_result_bs${BATCH_SIZE}_fl2.json"
        fi

        # 執行 python 指令
        python lab5.py \
            --batch_size ${BATCH_SIZE} \
			--seq_len ${SEQ_LEN} \
			--num_heads ${NUM_HEADS} \
			--emb_dim ${EMB_DIM} \
            --impl ${IMPL} \
            --causal \
            --repeats 30 \
            --output ${OUTPUT_FILE}
    done
done
