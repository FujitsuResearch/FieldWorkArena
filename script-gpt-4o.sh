#!/bin/bash

SCRIPT_DATE=$(date +%Y%m%d)
LLM_MODEL="gpt-4o"
PATH_GT="./Sample_Data/GT_V1"

bash run_tasks_icpr2026.sh factory   results_fwa_icpr2026_factory_${LLM_MODEL}_${SCRIPT_DATE}   ${LLM_MODEL}
bash run_tasks_icpr2026.sh warehouse results_fwa_icpr2026_warehouse_${LLM_MODEL}_${SCRIPT_DATE} ${LLM_MODEL}
bash run_tasks_icpr2026.sh retail    results_fwa_icpr2026_retail_${LLM_MODEL}_${SCRIPT_DATE}    ${LLM_MODEL}

sleep 10

cd tools/evaluation

# Evaluate results
python run-evaluator.py ../../results_fwa_icpr2026_factory_${LLM_MODEL}_${SCRIPT_DATE}   ${PATH_GT} results_fwa_icpr2026_factory_${LLM_MODEL}_${SCRIPT_DATE}.json
python run-evaluator.py ../../results_fwa_icpr2026_warehouse_${LLM_MODEL}_${SCRIPT_DATE} ${PATH_GT} results_fwa_icpr2026_warehouse_${LLM_MODEL}_${SCRIPT_DATE}.json
python run-evaluator.py ../../results_fwa_icpr2026_retail_${LLM_MODEL}_${SCRIPT_DATE}    ${PATH_GT} results_fwa_icpr2026_retail_${LLM_MODEL}_${SCRIPT_DATE}.json

sleep 10

# Output category-wise results
python calc_category.py ./Sample_Data/category_factory.csv results_fwa_icpr2026_factory_${LLM_MODEL}_${SCRIPT_DATE}.json_detail.txt > results_fwa_icpr2026_factory_${LLM_MODEL}_${SCRIPT_DATE}_category.txt
python calc_category.py ./Sample_Data/category_warehouse.csv results_fwa_icpr2026_warehouse_${LLM_MODEL}_${SCRIPT_DATE}.json_detail.txt > results_fwa_icpr2026_warehouse_${LLM_MODEL}_${SCRIPT_DATE}_category.txt
python calc_category.py ./Sample_Data/category_retail.csv results_fwa_icpr2026_retail_${LLM_MODEL}_${SCRIPT_DATE}.json_detail.txt > results_fwa_icpr2026_retail_${LLM_MODEL}_${SCRIPT_DATE}_category.txt

cd ../..



