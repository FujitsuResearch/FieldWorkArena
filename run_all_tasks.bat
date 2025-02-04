@echo off
for /F "tokens=*" %%i in (all_task_ids.txt) do (
    python demo/run.py --task_name %%i --result_dir results/
)