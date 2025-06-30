@echo off
REM Usage: run_tasks.bat <task>
REM <task>: all / factory / warehouse / retail

set TASK=%1

if "%TASK%"=="all" (
    echo Running all tasks...
    for /F "tokens=*" %%i in (all_task_ids_factory.txt) do (
        python3 demo/run.py --task_name %%i --result_dir results/
    )
    for /F "tokens=*" %%i in (all_task_ids_warehouse.txt) do (
        python3 demo/run.py --task_name %%i --result_dir results/
    )
    for /F "tokens=*" %%i in (all_task_ids_retail.txt) do (
        python3 demo/run.py --task_name %%i --result_dir results/
    )
) else if "%TASK%"=="factory" (
    echo Running factory tasks...
    for /F "tokens=*" %%i in (all_task_ids_factory.txt) do (
        python3 demo/run.py --task_name %%i --result_dir results/
    )
) else if "%TASK%"=="warehouse" (
    echo Running warehouse tasks...
    for /F "tokens=*" %%i in (all_task_ids_warehouse.txt) do (
        python3 demo/run.py --task_name %%i --result_dir results/
    )
) else if "%TASK%"=="retail" (
    echo Running retail tasks...
    for /F "tokens=*" %%i in (all_task_ids_retail.txt) do (
        python3 demo/run.py --task_name %%i --result_dir results/
    )
) else (
    echo Usage: run_tasks.bat <task>
    echo <task>: all / factory / warehouse / retail
)

pause