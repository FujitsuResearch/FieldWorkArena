import logging
import pathlib
import json
from typing import List, Optional, Tuple

import numpy as np
import playwright.sync_api

from browsergym.core.task import AbstractBrowserTask  # Correct import
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from browsergym.workarena.config import SNOW_JS_UTILS_FILEPATH
from browsergym.workarena.utils import url_login
from browsergym.workarena.api.user import create_user

from ...config import DATA_PATH

import base64
import cv2
import os

from ...metrics.automatic.automatic_evaluation import llm_fuzzy_match


def process_video(video_path, seconds_per_frame=2):
    logging.info("Processing video: %s", video_path)
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info("Total frames: %s", total_frames)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    logging.info("Number of frames: %s", len(base64Frames))
    video.release()

    return base64Frames

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _build_goal_and_answer(config, with_na_hint = False):
    goal_text = "Answer the following question based on the provided image. \n"
    if with_na_hint:
        goal_text += """\
        If you believle the task is impossible, to complete, provide the answer "N/A" \n.
        """
    
    query = config["conversations"][0]["value"]
    answer = config["conversations"][1]["value"]

    goal_text += query

    goal = [{"type": "text", "text": goal_text}]

    data_type, data = config["input_data"].split(" ", 1)
    data_type = data_type.strip()
    data = data.strip()

    data = os.path.join(DATA_PATH,  data)
    if data_type == "image":
        base64_image = encode_image(data)
        goal.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"})
    if data_type =="Video":
        base64_frames = process_video(data)

        for frame in base64_frames:
            goal.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame}"})
            
    return goal, answer

class GenericGroup2Task(AbstractServiceNowTask):  # Inherit from AbstractServiceNowTask
    def __init__(
            self,
            seed: Optional[int] = None,
            task_id: Optional[str] = None,
            instance = None, # Add instance argument
            start_rel_url: str = "/now/nav/ui/classic/", # Add start_rel_url
            final_rel_url: Optional[str] = None, # Add final_rel_url
            user_roles: List[str] = ["admin"], # Add user_roles
            has_description: bool = False, # Add has_description
            **kwargs,    
    ) -> None:
        # Call super().__init__ with necessary arguments
        super().__init__(seed=seed, instance=instance, start_rel_url=start_rel_url, final_rel_url=final_rel_url, user_roles=user_roles, has_description=has_description)

        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.config_file: str = None

        config_dir = pathlib.Path(__file__).parent
        all_configs_str = ""

        for config_file in config_dir.glob("*.json"):
            with open(config_file, 'r', encoding='utf-8') as f:
                all_configs_str += f.read()

        all_configs = json.loads(all_configs_str)

        self.used_in_level_2 = True

        if task_id is not None:
            self.task_configs = [config for config in all_configs if config["id"] == task_id]
        else:
            self.task_configs = all_configs
        self.is_validated = True
        self.__dict__.update(kwargs)

    def setup_goal(self, page: playwright.sync_api.Page) -> Tuple[str, dict]: # Implement abstract method
        self.config = self.random.choice(self.task_configs)
        #self.is_validated = True
        self.goal, self.answer = _build_goal_and_answer(self.config)

        page.context.set_geolocation(None)

        if self.config.get("start_url"): # Use get to handle potential missing key
            with page.expect_navigation(): # Add expect_navigation for reliability
                page.goto(str(self.config["start_url"])) # Convert Path to string

        return self.goal, {}


    @classmethod
    def get_task_id(cls) -> str:
        return "generic_group_2_task"  # Provide a task ID

    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        # Implement cheat method or raise NotImplementedError if cheating is not supported
        raise NotImplementedError("Cheat function not implemented for this task.")

    def validate(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> Tuple[float, bool, str, dict]:
        try:
            if chat_messages[-1]["role"] == "assistant":
                score = llm_fuzzy_match(self.answer, chat_messages[-1]["message"], self.goal[0], disallow_partial=False)
            else:
                return 0.0, True, "", {"message": "No message from assistant"}
            if score > 0.5:
                return 1.0, True, "Correct", {}
            else:
                return 0.0, True, "", {"message": "Incorrect"}
        except IndexError: # Handle cases where chat_messages is empty
            return 0.0, False, "", {}

