#!/bin/bash

# Usage: run_tasks.sh <task>
# <task> all / factory / warehouse / retail

TASK="$1"
RESULTS_PATH="$2"
MODEL="$3"
VLM="$4"
SCRIPT_PATH=demo_fwa_icpr2026


# Define the script to run and the input file based on the task
case "$TASK" in
  all)
    SCRIPT="
    while IFS= read -r line; do
        python3 ${SCRIPT_PATH}/run.py --task_name \$line --result_dir ${RESULTS_PATH}/ --model_name ${MODEL} 
    done < all_task_ids_factory.txt

    while IFS= read -r line; do
        python3 ${SCRIPT_PATH}/run.py --task_name \$line --result_dir ${RESULTS_PATH} --model_name ${MODEL} 
    done < all_task_ids_warehouse.txt

    while IFS= read -r line; do
        python3 ${SCRIPT_PATH}/run.py --task_name \$line --result_dir ${RESULTS_PATH} --model_name ${MODEL} 
    done < all_task_ids_retail.txt
    "
    ;;
  factory)
    SCRIPT="
    while IFS= read -r line; do
        python3 ${SCRIPT_PATH}/run.py --task_name \$line --result_dir ${RESULTS_PATH} --model_name ${MODEL} 
    done < all_task_ids_factory.txt
    "
    ;;
  warehouse)
    SCRIPT="
    while IFS= read -r line; do
        python3 ${SCRIPT_PATH}/run.py --task_name \$line --result_dir ${RESULTS_PATH} --model_name ${MODEL}
    done < all_task_ids_warehouse.txt
    "
    ;;
  retail)
    SCRIPT="
    while IFS= read -r line; do
        python3 ${SCRIPT_PATH}/run.py --task_name \$line --result_dir ${RESULTS_PATH} --model_name ${MODEL} 
        sleep 30
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
