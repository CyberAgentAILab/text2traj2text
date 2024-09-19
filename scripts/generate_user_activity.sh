#! /bin/bash

docker exec text2traj2text python3 scripts/text2traj/generate_user_captions.py
docker exec text2traj2text python3 scripts/text2traj/generate_purchase_list.py
docker exec text2traj2text python3 scripts/text2traj/generate_paraphrasing.py
docker exec text2traj2text python3 scripts/text2traj/generate_trajectory.py