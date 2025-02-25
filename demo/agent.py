import base64
import dataclasses
import io
import logging

import numpy as np
import openai
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.action.python import PythonActionSet
from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from copy import deepcopy
from langchain_community.document_loaders import PyPDFLoader


import re
import cv2


logger = logging.getLogger(__name__)

def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


# for feeding video frames to OpenAI API
def process_video(video_path, seconds_per_frame=1):
    logging.info("Processing video: %s", video_path)
    base64Frames = []
    max_frames = 30

    video = cv2.VideoCapture(video_path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info("Total frames: %s", total_frames)

    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    if frames_to_skip < total_frames / (max_frames - 1):
        frames_to_skip = int(total_frames / (max_frames - 1))
    while curr_frame < total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        base64Frames.append(image_to_jpg_base64_url(frame))
        curr_frame += frames_to_skip
    seconds_per_frame = frames_to_skip / fps
    logging.info("Number of frames: %s", len(base64Frames))
    video.release()

    return base64Frames, seconds_per_frame

def parse_goal_object(goal_object):
    goals = []#deepcopy(goal_object)
    image_paths = set()
    video_paths = set()
    document_paths = set()
    text_paths = set()
    
    for msg in goal_object:
        if not msg["type"] == "text":
            continue
        tags = re.findall("\n.*?.jpg\n", msg["text"])
        for tag in tags:
            image_paths.add(tag.strip())
        tags = re.findall("\n.*?.mp4\n", msg["text"])
        for tag in tags:
            video_paths.add(tag.strip())
        tags = re.findall("\n.*?.pdf\n", msg["text"])
        for tag in tags:
            document_paths.add(tag.strip())
        tags = re.findall("\n.*?.txt\n", msg["text"])
        for tag in tags:
            text_paths.add(tag.strip())

    
    #if not image_paths == set():
    for i, image_path in enumerate(image_paths):
        if i == 0:
            goals.append({"type": "text", "text": "\nThese are the images you are provided."})
        goals.append({"type": "text", "text": image_path + ": "})
        base64_image = image_to_jpg_base64_url(Image.open(image_path))
        goals.append({"type": "image_url", "image_url": {'url': base64_image, 'detail': 'auto'}})

    for i, video_path in enumerate(video_paths):
        if i == 0:
            goals.append({"type": "text", "text": "\nYou cannot see video directly, so you MUST use these frames decimated from the video."})
        goals.append({"type": "text", "text": video_path + ": "})
        base64Frames, seconds_per_frame = process_video(video_path)
        if len(base64Frames) == 0:
            return None
        
        #timestamps = [float(i * frames_to_skip) / 30 for i in range(len(base64Frames))]

        for i, base64_frame in enumerate(base64Frames):
            goals.append({"type": "text", "text": f"timestamp: {i * seconds_per_frame:.2f} [s]"})
            goals.append({"type": "image_url", "image_url": {'url': base64_frame, 'detail': 'auto'}})

    # only use text from pdf
    for i, document_path in enumerate(document_paths):
        if i == 0:
            goals.append({"type": "text", "text": "\nThese are the text extracted from the document you are provided."})
        goals.append({"type": "text", "text": document_path + ": "})
        loader = PyPDFLoader(document_path)
        document = loader.load()
        
        #print(document_text)
        document_text = ""
        for page in document:
            document_text += page.page_content + "\n"
        goals.append({"type": "text", "text": document_text})

    for i, text_path in enumerate(text_paths):
        if i == 0:
            goals.append({"type": "text", "text": "\nThese are the text you are provided."})
        goals.append({"type": "text", "text": text_path + ": "})
        with open(text_path, "r") as f:
            text = f.read()
        goals.append({"type": "text", "text": text})
    return goals

class DemoAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    def obs_preprocessor(self, obs: dict) -> dict:

        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        self.openai_client = openai.OpenAI()

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],  # define a subset of the action space
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode=demo_mode,  # add visual effects
        )
        # use this instead to allow the agent to directly use Python code
        # self.action_set = PythonActionSet())

        self.action_history = []
        self.loaded_data = ""

    def get_action(self, obs: dict) -> tuple[str, dict]:
        system_msgs = []
        user_msgs = []

        if self.chat_mode:
            system_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Instructions
You are an AI assistant, your goal is to help the user to achieve their goal.
You can communicate with the user via chat and see images provided by the user.
You have access to a web browser that both you and the user can see, and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
                }
            )
            # append chat messages
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Chat Messages
""",
                }
            )
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
- [{msg['role']}] {msg['message']}
""",
                        }
                    )
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

        else:
            assert obs["goal_object"], "The goal is missing."
            system_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
                }
            )
            # append goal
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Goal
""",
                }
            )
            # goal_object is directly presented as a list of openai-style messages
            #print(self.loaded_data)
            if self.loaded_data == "":
                self.loaded_data = parse_goal_object(obs["goal_object"])
            
            user_msgs.extend(obs["goal_object"])
            user_msgs.extend(self.loaded_data)

        # append url of all open tabs
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Currently open tabs
""",
            }
        )
        for page_index, (page_url, page_title) in enumerate(
            zip(obs["open_pages_urls"], obs["open_pages_titles"])
        ):
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
  Title: {page_title}
  URL: {page_url}
""",
                }
            )

        # append page AXTree (if asked)
        if self.use_axtree:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Current page Accessibility Tree

{obs["axtree_txt"]}

""",
                }
            )
        # append page HTML (if asked)
        if self.use_html:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Current page DOM

{obs["pruned_html"]}

""",
                }
            )

        # append page screenshot (if asked)
        if self.use_screenshot:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
# Current page Screenshot
""",
                }
            )
            user_msgs.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot"]),
                        "detail": "auto",
                    },  # Literal["low", "high", "auto"] = "auto"
                }
            )

        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

You should first click 'All' button.
""",
            }
        )

        # append past actions (and last error message) if any
        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# History of past actions
""",
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\

{action}
""",
                    }
                    for action in self.action_history
                ]
            )

            if obs["last_action_error"]:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
# Error message from last action

{obs["last_action_error"]}

""",
                    }
                )

        # ask for the next action
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.
You MUST answer with a single action.
""",
            }
        )

        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            match message["type"]:
                case "text":
                    
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append(
                            "image_url: " + image_url[:30] + "... (truncated)"
                        )
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(
                        f"Unknown message type {repr(message['type'])} in the task goal."
                    )
        full_prompt_txt = "\n".join(prompt_text_strings)
        logger.info(full_prompt_txt)

        # query OpenAI model
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msgs},
                {"role": "user", "content": user_msgs},
            ],
        )
        action = response.choices[0].message.content

        self.action_history.append(action)

        return action, {}


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    model_name: str = "gpt-4o-mini"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
        )
