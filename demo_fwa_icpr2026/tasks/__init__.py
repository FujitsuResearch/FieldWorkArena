import os
import json
import logging

from browsergym.core.registration import register_task
from .fwa_task import GenericGroup2Task, JSONOutputTask

import logging

logger = logging.getLogger(__name__)

# tmp
TASK_PATH = "./benchmark/tasks"

G2_TASK_PATH = os.path.join(TASK_PATH, "group2")

DATA_DIR = "./data/"
IMAGE_DIR = DATA_DIR + "image/"
MOVIE_DIR = DATA_DIR + "movie/"
DOC_DIR = DATA_DIR + "document/"


ALL_FIELDWORKARENA_TASK_IDS = []

# Add all tasks in group2 to the registry
for task_file in os.listdir(G2_TASK_PATH):
    if task_file.endswith(".json"):
        # Read the JSON file
        with open(os.path.join(G2_TASK_PATH, task_file), 'r', encoding='utf-8') as f:
            task_configs = json.load(f)
            for task_config in task_configs:
                task_id = "fieldworkarena." + task_config["id"]
                #print(task_id)
                if task_config["output_format"] == "json":
                    register_task(task_id, JSONOutputTask, task_kwargs={"task_id": task_id})
                else:
                    register_task(task_id, GenericGroup2Task, task_kwargs={"task_id": task_id})
                ALL_FIELDWORKARENA_TASK_IDS.append(task_id)

#register_task(id=SampleTask.get_task_id(), task_class=SampleTask)
#print(f'task {SampleTask.get_task_id()} is registered.')