from importlib import resources
import os
from benchmark import tasks

# tmp
TASK_PATH = str(resources.files(tasks))

#G1_TASK_PATH = str
#G2_TASK_PATH = os.path.join(TASK_PATH, "group2")
G2_TASK_PATH = str(resources.files(tasks).joinpath("group2"))
#G3_TASK_PATH = os.path.join(TASK_PATH, "group3")

DATA_PATH = "./data/raw/"