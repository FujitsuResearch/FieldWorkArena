__version__ = "0.1.0"

import os
from browsergym.core.registration import register_task
import gymnasium as gym
import json
from . import config
from .tasks.group2.group2_tasks import GenericGroup2Task
from .tasks.long_horizon.see_image_and_do import SeeImageAndDoTask, SeeImageAndCreateIncidentTask

ALL_FIELDWORKARENA_TASK_IDS = []

# Add all tasks in group2 to the registry
for task_file in os.listdir(config.G2_TASK_PATH):
    if task_file.endswith(".json"):
        #jsonファイルを読み込む
        with open(os.path.join(config.G2_TASK_PATH, task_file), 'r', encoding='utf-8') as f:
            task_configs = json.load(f)
            for task_config in task_configs:
                register_task(task_config["id"], GenericGroup2Task, task_kwargs={"task_id": task_config["id"]})
                #ALL_FIELDWORKARENA_TASK_IDS.append(task_config["id"])
                report_task_id = task_config["id"] + ".report"
                register_task(report_task_id, SeeImageAndCreateIncidentTask, task_kwargs={"image_task_id": task_config["id"]})
