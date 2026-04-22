import base64
import dataclasses
import io
import logging
import json
import os
import time

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

from llm import GeminiAccessor, ChatGPTAccessor, OllamaAccessor


from dotenv import load_dotenv
load_dotenv(override=True)

from actions import analyze_video, analyze_image

USE_QWEN3 = False

if USE_QWEN3:
    from analyzer_qwen3 import query_video_qwen3, query_image_qwen3

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
    THRESHOLD=250000

    if frames_to_skip < total_frames / (max_frames - 1):
        frames_to_skip = int(total_frames / (max_frames - 1))
    while curr_frame < total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        quality = 90
        is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        while is_success and len(buffer) > THRESHOLD:
            quality -= 5
            if quality < 10:
                is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                break
            else:
                is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        logger.info("buffer size: %d", len(buffer))
        base64frame = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
        base64Frames.append(base64frame)
        #base64Frames.append(image_to_jpg_base64_url(frame))
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
        vlm_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.vlm_name = vlm_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.analyzed = dict()

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        # LLM setup
        if self.model_name.startswith("gpt"):
            self.client = ChatGPTAccessor(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            logger.info("Using OpenAI API with model: %s", self.model_name)
        elif self.model_name.startswith("gemini"):
            self.client = GeminiAccessor(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("GEMINI_BASE_URL"),
            )
            logger.info("Using Gemini API with model: %s", self.model_name)
        elif self.model_name.startswith("claude"):
            self.client = ChatGPTAccessor(
                api_key=os.getenv("CHATAI_GPT_API_KEY"),
                base_url=os.getenv("CHATAI_GPT_BASE_URL"),
            )
            logger.info("Using Claude API with model: %s", self.model_name)
        elif self.model_name.startswith("llama"):
            self.client = OllamaAccessor(
                base_url=os.getenv("OLLAMA_BASE_URL"),
            )
            logger.info("Using Llama(ollama) API with model: %s", self.model_name)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # VLM (Vision Language Model) setup
        if self.vlm_name is None:
            self.client_vlm = self.client
            self.vlm_name = self.model_name
            logger.info("No VLM specified, using the same model for vision tasks.")
        else:
            # Initialize a different client for VLM if specified
            if self.vlm_name.startswith("gpt"):
                self.client_vlm = ChatGPTAccessor(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                )
                logger.info("Using OpenAI API for VLM: %s", self.vlm_name)
            elif self.vlm_name.startswith("gemini"):
                self.client_vlm = GeminiAccessor(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    base_url=os.getenv("GEMINI_BASE_URL"),
                )
                logger.info("Using Gemini API for VLM: %s", self.vlm_name)
            elif self.vlm_name.startswith("claude"):
                self.client_vlm = ChatGPTAccessor(
                    api_key=os.getenv("CLAUDE_API_KEY"),
                    base_url=os.getenv("CLAUDE_BASE_URL"),
                )
                logger.info("Using Claude API for VLM: %s", self.vlm_name)
            elif self.vlm_name.startswith("llama") or self.vlm_name.startswith("qwen"):
                self.client_vlm = OllamaAccessor(
                    base_url=os.getenv("OLLAMA_BASE_URL"),
                )
                logger.info("Using Llama(ollama) API for VLM: %s", self.vlm_name)
                logger.info(f"base url: {os.getenv('OLLAMA_BASE_URL')}")
            else:
                raise ValueError(f"Unsupported VLM name: {self.vlm_name}")

        self.action_set = HighLevelActionSet(
            #subsets=["chat", "tab", "nav", "bid", "infeas"],  # define a subset of the action space
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            subsets=["chat", "custom"],
            custom_actions=[analyze_video, analyze_image],
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode=demo_mode,  # add visual effects
        )
        # use this instead to allow the agent to directly use Python code
        # self.action_set = PythonActionSet())

        self.action_history = []
        self.loaded_data = ""


    def get_action(self, obs: dict) -> tuple[str, dict]:
        #logger.info(f'Getting action from the agent with observation: {obs}')
        system_msgs = []
        user_msgs = []
        contents_paths = {}

        for msg in obs["goal_object"]:
            if not msg["type"] == "text":
                continue
            tags = re.findall("\n.*?.jpg\n", msg["text"])
            for tag in tags:
                tag = tag.strip()
                contents_paths[os.path.basename(tag)] = tag
                if tag not in self.analyzed:
                    self.analyzed[tag] = "Not analyzed yet."
            tags = re.findall("\n.*?.mp4\n", msg["text"])
            for tag in tags:
                tag = tag.strip()
                contents_paths[os.path.basename(tag)] = tag
                if tag not in self.analyzed:
                    self.analyzed[tag] = "Not analyzed yet."


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

Review the instructions from the user and all other information to find the best possible next action to accomplish your goal.
Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
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

        #
        # action space for image/video
        #

        if obs["chat_messages"][-1]['message'].startswith("analyze_image"):
            _, image_path, query = obs["chat_messages"][-1]['message'].split(",", 2)
            logger.info(f"Analyzing image... {image_path}")
            # revise file path
            image_path = os.path.basename(image_path)
            if image_path in contents_paths:
                image_path = contents_paths[image_path]


            if USE_QWEN3:
                action = query_image_qwen3(image_path, query)
                result = True

            else:

                user_prompt = self.client_vlm.process_vision(image_paths=[image_path])
            
                action = self.client_vlm.generate_response(
                    model=self.vlm_name,
                    system_prompt=[query],
                    user_prompt=user_prompt,
                    temperature=0,
                )
            result = True
            logger.info(f"Image analysis response: {action}")

            if result is True:
                if image_path in self.analyzed:
                    self.analyzed[image_path] = action


        if obs["chat_messages"][-1]['message'].startswith("analyze_video"):
            _, video_path, query = obs["chat_messages"][-1]['message'].split(",", 2)
            # revise file path
            video_path = os.path.basename(video_path)
            if video_path in contents_paths:
                video_path = contents_paths[video_path]
                logger.info(f"Analyzing video... {video_path}")

            if video_path in self.analyzed:

                if USE_QWEN3:
                    action = query_video_qwen3(video_path, query)
                    result = True

                else:

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_length = total_frames / fps
                    cap.release()

                    user_prompt = self.client_vlm.process_vision(video_paths=[video_path])

                    action = self.client_vlm.generate_response(
                        model=self.vlm_name,
                        system_prompt=[query],
                        user_prompt=user_prompt,
                        temperature=0,
                    )
                    logger.info(f"Video analysis response: {action}")
                    result = True
            
                if result is True:
                    self.analyzed[video_path] = action

        analyzed = []
        for key, value in self.analyzed.items():
            analyzed.append(f"{key}: {value}")

        analyzed = "\n\n".join(analyzed)

        # Analyzed video
        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Analyzed Results

{analyzed}

""",
            }
        )



        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Action Space

{self.action_set.describe(with_long_description=True, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

I need to analyze the image provided by the user to find out how many workers.
```analyze_scene_image("scene1.jpg", "Count the number of workers in the image.")```


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
The action MUST use the format defined above.
""",
            }
        )

        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            if 'type' not in message.keys():
                break
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
        #logger.info(f'system_msgs: {system_msgs}')
        #logger.info(f'user_msgs: {user_msgs}')

        action = self.client.generate_response(
            model=self.model_name,
            system_prompt=system_msgs,
            user_prompt=user_msgs,
            temperature=0,
        )

        self.action_history.append(action)
        logger.info(f"tokens: {self.client.get_token_usage()}")

        time.sleep(10)
        return action, {}


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    model_name: str = "gpt-4o-mini"
    vlm_name: str = None
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            vlm_name=self.vlm_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
        )