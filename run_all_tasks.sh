#!/bin/bash
while IFS= read -r line; do
    python demo/run.py --task_name $line --result_dir results/
done < all_task_ids.txt