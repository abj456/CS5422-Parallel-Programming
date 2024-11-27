#!/bin/bash
IMPLS=("Pytorch" "Flash2")

# 變動參數: BATCH_SIZE
BATCH_SIZE=32
SEQ_LENS=(128 256 512)
NUM_HEADS=32
EMB_DIM=2048
DIR_NAME="seq_len"

# 迴圈來變化 BATCH_SIZE
for SEQ_LEN in "${SEQ_LENS[@]}"
do
    for IMPL in "${IMPLS[@]}"
    do
        # 根據 IMPL 設定不同的輸出文件名
        if [ "$IMPL" == "Pytorch" ]; then
            OUTPUT_FILE="${DIR_NAME}/benchmark_result_sl${SEQ_LEN}_pth.json"
        else
            OUTPUT_FILE="${DIR_NAME}/benchmark_result_sl${SEQ_LEN}_fl2.json"
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
