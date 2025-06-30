#!/bin/bash

# Usage: run_tasks.sh <task>
# <task> all / factory / warehouse / retail

TASK="$1"

# Define the script to run and the input file based on the task
case "$TASK" in
  all)
    SCRIPT="
    while IFS= read -r line; do
        python3 demo/run.py --task_name \$line --result_dir results/
    done < all_task_ids_factory.txt

    while IFS= read -r line; do
        python3 demo/run.py --task_name \$line --result_dir results/
    done < all_task_ids_warehouse.txt

    while IFS= read -r line; do
        python3 demo/run.py --task_name \$line --result_dir results/
    done < all_task_ids_retail.txt
    "
    ;;
  factory)
    SCRIPT="
    while IFS= read -r line; do
        python3 demo/run.py --task_name \$line --result_dir results/
    done < all_task_ids_factory.txt
    "
    ;;
  warehouse)
    SCRIPT="
    while IFS= read -r line; do
        python3 demo/run.py --task_name \$line --result_dir results/
    done < all_task_ids_warehouse.txt
    "
    ;;
  retail)
    SCRIPT="
    while IFS= read -r line; do
        python3 demo/run.py --task_name \$line --result_dir results/
    done < all_task_ids_retail.txt
    "
    ;;
  *)
    echo "Usage: $0 <task>"
    echo "  <task> all / factory / warehouse / retail"
    exit 1
    ;;
esac

# Execute the script
eval "$SCRIPT"
