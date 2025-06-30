import logging
import pathlib
import json
from typing import List, Optional, Tuple

import numpy as np
import playwright.sync_api
from json import JSONDecodeError

from browsergym.core.task import AbstractBrowserTask  # Correct import
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from browsergym.workarena.config import SNOW_JS_UTILS_FILEPATH
from browsergym.workarena.utils import url_login
from browsergym.workarena.api.user import create_user

from ...config import DATA_DIR, IMAGE_DIR, MOVIE_DIR, DOC_DIR

import base64
import cv2
import os

from ...metrics.automatic.automatic_evaluation import llm_fuzzy_match
import re

def _return_path(data_name):
    if data_name.endswith("jpg") or data_name.endswith("png"):
        return os.path.join(IMAGE_DIR, data_name)
    elif data_name.endswith("mp4"):
        return os.path.join(MOVIE_DIR, data_name)
    elif data_name.endswith("pdf") or data_name.endswith("txt"):
        return os.path.join(DOC_DIR, data_name)
    else:
        return os.path.join(DATA_DIR, data_name)

def _build_goal(config, with_na_hint = False, only_json_output = False):
    goal_text = "Answer the following question based on the provided file.\n"
    if only_json_output:
        goal_text = "Make string that can be parsed as JSON, and provide the stirng with send_msg_to_user action.\n"
    else:
        goal_text = "Give the answer with not report_infeasible but send_msg_to_user action.\n"
    if with_na_hint:
        goal_text += """\
        If you don't know the answer, you can type "I don't know" or "N/A".
        """
    
    query = config["conversations"][0]["value"]
    #answer = config["conversations"][1]["value"]

    goal_text += query

    if type(config["input_data"]) == str:
        data_type, data_path = config["input_data"].split(" ", 1)
        data_type = data_type.strip()
        data_path = data_path.strip()
        #data_path = os.path.join(DATA_DIR, data_path)
        goal_text = goal_text + "\nData is stored in [" + data_type + " " + data_path + "]\n\n"
    if type(config["input_data"]) == list:
        goal_text = goal_text + "\nData is stored in "
        for data_path in config["input_data"]:
            #print(data_path)
            data_path = _return_path(data_path)
            goal_text = goal_text + f"\n{data_path}\n"
            
    goal = [{"type": "text", "text": goal_text}]

            
    return goal#, answer

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
        all_configs = []

        for config_file in config_dir.glob("*.json"):
            with open(config_file, 'r', encoding='utf-8') as f:
                #all_configs_str += f.read()
                tmp = f.read()
                all_configs.extend(json.loads(tmp))

        self.used_in_level_2 = True

        if task_id is not None:
            self.task_configs = [config for config in all_configs if "fieldworkarena." +  config["id"] == task_id]
        else:
            self.task_configs = all_configs
        self.task_id = task_id 
        self.is_validated = True
        self.__dict__.update(kwargs)

    def setup_goal(self, page: playwright.sync_api.Page) -> Tuple[str, dict]: # Implement abstract method
        self.config = self.random.choice(self.task_configs)
        #self.is_validated = True
        self.goal = _build_goal(self.config)

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
                logging.info(f"\n<id>{self.task_id}</id>\n<answer>{chat_messages[-1]['message']}</answer>")

                return 1.0, True, "Recieved answer", {}
            else:
                return 0.0, False, "Give an answer in the chat", {}
        except IndexError: # Handle cases where chat_messages is empty
            return 0.0, False, "", {}



class JSONOutputTask(GenericGroup2Task):
    def setup_goal(self, page: playwright.sync_api.Page) -> Tuple[str, dict]:
        self.config = self.random.choice(self.task_configs)
        self.goal = _build_goal(self.config, only_json_output=True)

        page.context.set_geolocation(None)

        if self.config.get("start_url"):
            with page.expect_navigation():
                page.goto(str(self.config["start_url"]))

        return self.goal, {}

    def validate(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> Tuple[float, bool, str, dict]:
        try:
            if chat_messages[-1]["role"] == "assistant":
                # Extract JSON part from the message
                json_part = re.search(r'\{.*\}|\[.*\]', chat_messages[-1]["message"], re.DOTALL)
                if json_part:
                    json_str = json_part.group(0)
                    try:
                        json.loads(json_str)
                        # logging.info(f"\n<answer>\nid: {self.task_id} \n answer: {json_str}\n</answer>")
                        logging.info(f"\n<id>{self.task_id}</id>\n<answer>{json_str}</answer>")
                        return 1.0, True, "Correct format", {}
                    except JSONDecodeError:
                        return 0.0, False, "Answer with correct JSON format", {}
                else:
                    return 0.0, False, "Answer with correct JSON format", {}
            else:
                return 0.0, False, "Answer in chat", {"message": "No message from assistant"}
        except IndexError: # Handle cases where chat_messages is empty
            return 0.0, False, "", {}
