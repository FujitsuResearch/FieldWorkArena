import os
import logging
import json
import pathlib
from typing import List, Optional, Tuple

from json import JSONDecodeError

import playwright.sync_api
from browsergym.core.task import AbstractBrowserTask

import re

import tasks

logger = logging.getLogger(__name__)


def _return_path(data_name):
    if data_name.endswith("jpg") or data_name.endswith("png"):
        return os.path.join(tasks.IMAGE_DIR, data_name)
    elif data_name.endswith("mp4"):
        return os.path.join(tasks.MOVIE_DIR, data_name)
    elif data_name.endswith("pdf") or data_name.endswith("txt"):
        return os.path.join(tasks.DOC_DIR, data_name)
    else:
        return os.path.join(tasks.DATA_DIR, data_name)


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
            
    return goal


class GenericGroup2Task(AbstractBrowserTask):  # Inherit from AbstractBrowserTask
    def __init__(
            self,
            seed: Optional[int] = None,
            task_id: Optional[str] = None,
            config: Optional[dict] = None,
            **kwargs
        ) -> None:
        super().__init__(seed)
        self.task_id = task_id
        self.config = config
        #print('task_id', task_id)

        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.config_file: str = None

        #config_dir = pathlib.Path(__file__).parent
        config_dir = pathlib.Path(tasks.G2_TASK_PATH)
        
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

        #print(f"Task ID: {self.task_id}, Task Configs: {self.task_configs}")
        self.task_id = task_id 
        self.is_validated = True
        self.__dict__.update(kwargs)
        self.task_is_setup = False


    def setup_goal(self, page: playwright.sync_api.Page) -> Tuple[str, dict]: # Implement abstract method
        self.config = self.random.choice(self.task_configs)
        #self.is_validated = True
        self.goal = _build_goal(self.config)

        return self.goal, {}
    
    @classmethod
    def get_task_id(cls) -> str:
        return "generic_group_2_task"  # Provide a task ID

    def setup(self, page: playwright.sync_api.Page) -> Tuple[str, dict]:
        """
        Set up the task

        Parameters:
        -----------
        page: playwright.sync_api.Page
            The Playwright page object
        do_start: bool
            Whether to start the task or not (including navigating to start page) (default: True)

        """
        logging.debug("Setting up the base task")
        if self.task_is_setup:
            raise ValueError("The task is already setup")
        
        # Keep the page for client-side validation
        self.page = page

        goal, info = self.setup_goal(page=page)

        self.task_is_setup = True

        return goal, info

    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        # Implement cheat method or raise NotImplementedError if cheating is not supported
        raise NotImplementedError("Cheat function not implemented for this task.")

    def validate(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> Tuple[float, bool, str, dict]:
        try:
            if chat_messages[-1]['message'].startswith("analyze_video"):
                # Do nothing
                return 0.0, False, "", {}
            if chat_messages[-1]['message'].startswith("analyze_image"):
                # Do nothing
                return 0.0, False, "", {}
            if chat_messages[-1]["role"] == "assistant":
                logging.info(f"\n<id>{self.task_id}</id><answer>{chat_messages[-1]['message']}</answer>")

                return 1.0, True, "Recieved answer", {}
            else:
                return 0.0, False, "Give an answer in the chat", {}
        except IndexError: # Handle cases where chat_messages is empty
            return 0.0, False, "", {}

class JSONOutputTask(GenericGroup2Task):
    def __init__(
            self,
            seed: Optional[int] = None,
            task_id: Optional[str] = None,
            config: Optional[dict] = None,
            **kwargs
        ) -> None:
        super().__init__(seed, task_id, config, **kwargs)
    
    def setup_goal(self, page: playwright.sync_api.Page) -> Tuple[str, dict]:
        self.config = self.random.choice(self.task_configs)
        self.goal = _build_goal(self.config, only_json_output=True)

        return self.goal, {}

    def validate(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> Tuple[float, bool, str, dict]:
        try:
            if chat_messages[-1]['message'].startswith("analyze_video"):
                # Do nothing
                return 0.0, False, "", {}
            if chat_messages[-1]['message'].startswith("analyze_image"):
                # Do nothing
                return 0.0, False, "", {}
            if chat_messages[-1]["role"] == "assistant":
                # Extract JSON part from the message
                json_part = re.search(r'\{.*\}|\[.*\]', chat_messages[-1]["message"], re.DOTALL)
                if json_part:
                    json_str = json_part.group(0)
                    try:
                        json.loads(json_str)
                        # logging.info(f"\n<answer>\nid: {self.task_id} \n answer: {json_str}\n</answer>")
                        logging.info(f"\n<id>{self.task_id}</id><answer>{json_str}</answer>")
                        return 1.0, True, "Correct format", {}
                    except JSONDecodeError:
                        return 0.0, False, "Answer with correct JSON format", {}
                else:
                    return 0.0, False, "Answer with correct JSON format", {}
            else:
                return 0.0, False, "Answer in chat", {"message": "No message from assistant"}
        except IndexError: # Handle cases where chat_messages is empty
            return 0.0, False, "", {}

