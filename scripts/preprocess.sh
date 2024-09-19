DATASET_NAME=${1:-text2traj}

docker exec text2traj2text python3 src/text2traj2text/preprocess/preprocess.py \
    dataset_name=${DATASET_NAME} \
    test_size=0.2 \
    seed=42 \
    stay_duration_threshold=4 \
    human_stay_duration_threshold=10