from typing import Tuple
from browsergym.workarena.tasks.compositional.base import CompositionalTask
from browsergym.workarena.instance import SNowInstance
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from browsergym.workarena.tasks.navigation import AllMenuTask
from browsergym.workarena.tasks.form import CreateIncidentTask
from playwright.sync_api._generated import Page

from browsergym.workarena.api.utils import table_api_call

from ..group3.create_incident import CreateIncidentWithRetrievedInfoTask
from ..group2.group2_tasks import GenericGroup2Task
import numpy as np

class SeeImageAndDoTask(CompositionalTask):
    def __init__(
        self,
        seed: int = None,
        instance: SNowInstance = None,
        image_task_id: str = None,
        task: AbstractServiceNowTask = None,
    ) -> None:
        super().__init__(seed, instance)

        self.used_in_level_2 = True
        self.task = task
        self.task_description = ""
        self.short_description = "See image and provide the answer to the query in the chat. Then, perform the required action."
        self.image_task_id = image_task_id

        self.navigation_config={
            "application": "Service Desk",
            "module": "Incidents",
            "url": "/now/nav/ui/classic/params/target/incident_list.do",
        },
    
    def set_compositional_task(self) -> None:
        """
        Create and return the compositional task
        """
        raise NotImplementedError

    def get_compositional_task(self) -> list[AbstractServiceNowTask]:
        """
        Return the compositional task
        """
        return self.compositional_task

        
    def setup_goal(self, page: Page) -> tuple[str, dict]:
        self.set_compositional_task()
        config = self.fixed_config if self.fixed_config else self._get_config()
        super().setup_goal(page, [])


        self.valid_index = 0

        # Setup all the subtasks
        self.subtasks = []
        self.subgoals = []

        # Setup the first task
        self.subtasks.append(config[0])
        #for  task[]
        first_goal = self.subtasks[-1].setup(page, do_start=False)[0]
        encoded_images = first_goal[1:] 
        self.subgoals.append(first_goal[0]["text"])

        for task in config[1:]:
            self.subtasks.append(task)
            self.subgoals.append(self.subtasks[-1].setup(page=page, do_start=False)[0])

        task_intro = self.short_description + "\n"
        # Get the protocol to follow for the task and pre-pend it to the goal
        goal = task_intro
        goal += " \n Concretely, you need to complete the following steps:"

        i = 1
        for subgoal in self.subgoals:
            if not subgoal:
                continue
            goal += f"\n{i}. {subgoal}"
            i += 1
        oai_styled_goal = [{"type": "text", "text": goal}] + encoded_images

        return oai_styled_goal, {}

    
    def _get_config(self) -> list[AbstractServiceNowTask]:
        config = [
            GenericGroup2Task(
                seed=self.seed,
                instance= self.instance,
                task_id=self.image_task_id,
                is_validated =False,
            ),
            # navigation task
            AllMenuTask(
                instance=self.instance,
                fixed_config={
                    "application": "Service Desk",
                    "module": "Incidents",
                    "url": "/now/nav/ui/classic/params/target/incident_list.do",
                },
                is_validated=False,
                used_in_level_2=True,
            ),
        ] + self.get_compositional_task()

        return config
    


    def validate(self, page: Page, chat_messages: list[str]) -> Tuple[float, bool, str, dict]:
        super(CompositionalTask, self).validate(page, chat_messages)

        # Initialize the index of the first subtask that requires validation
        while (
            self.valid_index < len(self.subtasks) and not self.subtasks[self.valid_index].is_validated
        ):
            self.valid_index += 1
        if self.valid_index == len(self.subtasks):
            return (
                1,
                True,
                "Nice work, thank you!",
                {"message": "Task completed successfully."},
            )
        # Validate the current subtask
        subtask = self.subtasks[self.valid_index]
        #if not chat_messages['role'] == "user_image":
        reward, stop, info, message = subtask.validate(page, chat_messages)
        # If the subtask is valid
        if reward >= 1.0:
            # ... override the info and message to avoid success messages from the subtask
            info = message["message"] = (
                f"Step {self.valid_index + 1} has been completed successfully."
            )
            # ... this is a subtask, so we don't want to stop
            stop = False
            # ... increment index to flag this one as solved
            self.valid_index += 1
        # If the subtask is not valid
        else:
            # ... contextualize the info and message per subtask
            info = f"Step {self.valid_index + 1}: " + info
            message["message"] = f"Step {self.valid_index + 1}: " + message.get("message", "")
        # Check if all subtasks are solved
        if self.valid_index == len(self.subtasks):
            return (
                1,
                True,
                "Nice work, thank you!",
                {"message": "Task completed successfully."},
            )
        
        return 0, stop, info, message


class SeeImageAndCreateIncidentTask(SeeImageAndDoTask):
    def __init__(
            self, 
            seed: int = None, 
            instance: SNowInstance = None, 
            image_task_id: str = None) -> None:
        
        super().__init__(
            seed, 
            instance, 
            image_task_id=image_task_id,
        )
        self.task_description = "Create a new incident report based on the information you found in the image."
        self.task_short_description = "Create an incident report if you think it is necessary."

    def set_compositional_task(self) -> None:
        # temporary incident config
        import random

        base_user = table_api_call(
            instance=self.instance,
            table="sys_user",
            params={
                "sysparm_query": f"sys_id={self._base_user_sysid}",
            },
        )["result"][0]
        self.user_name = base_user["first_name"] + " " + base_user["last_name"]

        incident_number = "INC" + str(random.randint(1000000, 9999999))

        agent_full_name = "Bud Richman"

        incident_config = {
            "fields": {
                "caller_id": "Caller",
                "short_description": "Short description",
                "impact": "Impact",
                "number": "Number",
                "urgency": "Urgency",
            },
            "task_fields": [
                "caller_id",
                "short_description",
                "impact",
                "number",
                "urgency",
            ],
            "template_record": {
                "caller_id": self.user_name,
                "impact": "1 - High",
                "number": incident_number,
                "urgency": "1 - High",
            },
            "retrieve_fields": [
                "short_description",
            ],
        }

        self.compositional_task = [
            #CreateIncidentTask(
            CreateIncidentWithRetrievedInfoTask(
                instance = self.instance,
                fixed_config=incident_config,
                is_validated=True,
                used_in_level_2=True,
                check_record_created=False,            
            )
        ]
    def validate(self, page: Page, chat_messages: list[str]) -> Tuple[float, bool, str, dict]:
        return super().validate(page, chat_messages)