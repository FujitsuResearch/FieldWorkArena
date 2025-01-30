from importlib import resources
import os
from benchmark import tasks

# tmp
TASK_PATH = str(resources.files(tasks))

G2_TASK_PATH = str(resources.files(tasks).joinpath("group2"))

DATA_DIR = "./data/"
IMAGE_DIR = DATA_DIR + "image/"
MOVIE_DIR = DATA_DIR + "movie/"
DOC_DIR = DATA_DIR + "document/"
