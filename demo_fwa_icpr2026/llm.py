import os
import io
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple
import base64
import copy
import time

from dotenv import load_dotenv
load_dotenv(override=True)

import cv2
from PIL import Image

import openai
from google import genai


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# for feeding video frames to OpenAI API
def process_video(video_path: str, seconds_per_frame: float = 1.0) -> Tuple[List[str], float]:
    logging.info("Processing video: %s", video_path)
    base64Frames = []
    max_frames = 30

    video = cv2.VideoCapture(video_path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    logging.info(f"Total frames: {total_frames}, FPS: {fps}")

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
        base64frame = base64.b64encode(buffer).decode('utf-8')
        base64Frames.append(base64frame)
        #base64Frames.append(image_to_jpg_base64_url(frame))
        curr_frame += frames_to_skip
    seconds_per_frame = frames_to_skip / fps
    logging.info("Number of frames: %s", len(base64Frames))
    video.release()


    return base64Frames, seconds_per_frame


class LLMAccessor(ABC):
    """
    Abstract base class for accessing Large Language Models (LLMs).
    """
    def __init__(self, api_key: str = None, base_url: str = None):
        self._client = None
        self._input_tokens = 0
        self._output_tokens = 0


    def get_token_usage(self) -> Tuple[int, int]:
        """
        Returns the total number of input and output tokens used in interactions with the LLM.

        Returns:
            Tuple[int, int]: A tuple containing the total input tokens and total output tokens.
        """
        return self._input_tokens, self._output_tokens

    @abstractmethod
    def process_vision(self, image_paths: List[str] = [], video_paths: List[str] = [] ) -> dict:
        """
        Processes visual data (images and videos) and prepares it for LLM input.

        Args:
            image_paths (List[str]): List of file paths to images.
            video_paths (List[str]): List of file paths to videos.
        Returns:
            dict: A dictionary containing processed visual data ready for LLM input.

        """
        raise NotImplementedError("The process_vision method must be implemented by subclasses.")


    @abstractmethod
    def generate_response(self, model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generates a response from the LLM based on the given prompt.
        Concrete implementation is done in subclasses.

        Args:
            model (str): The model identifier to use for the LLM.
            system_prompt (str): The system prompt to provide to the LLM.
            user_prompt (str): The user prompt to provide to the LLM.
            **kwargs: Additional LLM-specific parameters (e.g., temperature, max_tokens).

        Returns:
            str: The response text from the LLM.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("The generate_response method must be implemented by subclasses.")


class ChatGPTAccessor(LLMAccessor):
    """
    Provides access to ChatGPT (OpenAI GPT models).
    """
    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key, base_url)
        #if api_key is None:
        #    api_key = os.getenv("OPENAI_API_KEY")
        #if base_url is None:
        #    base_url = os.getenv("OPENAI_BASE_URL")
        if api_key is None:
            raise ValueError("OpenAI API key and base URL must be set.")

        # Configure OpenAI client
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self._chunk_description_prompt="You are an expert on Video Analysis. You will be shown a series of images from a video and a goal. To derive the goal, describe what is happening in the video, including the objects, actions, and any other relevant details. Be as specific and detailed as possible."
        self._caption_summarization_prompt="Provide a comprehensive summary of the video based on the provided descriptions and a goal."

        logger.info("Initialized ChatGPTAccessor with base URL: %s", base_url)

    def process_vision(self, image_paths: List[str] = [], video_paths: List[str] = [] ) -> dict:
        """
        Processes visual data (images and videos) and prepares it for LLM input.

        Args:
            image_paths (List[str]): List of file paths to images.
            video_paths (List[str]): List of file paths to videos.
        Returns:
            dict: A dictionary containing processed visual data ready for LLM input.

        """
        logging.info("Processing visual data for LLM input.")
        results = []

        for image_path in image_paths:
            image = Image.open(image_path)

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

                results.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "low"
                    }
                })  

        # timestamp helper
        def seconds_to_hhmmss(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"


        for video_path in video_paths:
            base64Frames, seconds_per_frame = process_video(video_path)

            for i, frame_base64 in enumerate(base64Frames):
                timestamp = seconds_to_hhmmss(i * seconds_per_frame)
                results.append({
                    "type": "text",
                    "text": f"timestamp: {timestamp}"
                    })
                results.append({
                    "type": "image_url",
                    "image_url": {
                        'url': f"data:image/jpeg;base64,{frame_base64}",
                        'detail': 'low'
                    }
                })

        return results


    def generate_response(self, model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Sends a prompt to ChatGPT and generates a response.

        Args:
            system_prompt (str): The system prompt to provide to the LLM.
            user_prompt (str): The user prompt to provide to the LLM.
            model (str): The model identifier to use for the LLM.
            **kwargs: Additional parameters to pass to the OpenAI ChatCompletion API
                      (e.g., temperature, max_tokens, top_p, frequency_penalty, presence_penalty).

        Returns:
            str: The response text from ChatGPT.

        Raises:
            openai.error.OpenAIError: If an error occurs during the OpenAI API call.
        """
        if not self._client:
            raise ValueError("OpenAI client is not initialized.")
        logger.info(f"Generating response using model: {model}")
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs
            )
            self._input_tokens += response.usage.prompt_tokens
            self._output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content.strip()
        except openai.APIError as e:
            print(f"ChatGPT API error occurred: {e}")
            raise # Re-raise the error to allow the caller to handle it


    def _split_video_into_chunks(self, video_path: str, frames_per_second: float = 1.0, chunk_interval: int = 30) -> iter:
        """
        Splits a video into smaller segments (chunks) and saves them to a specified output directory.

        Args:

        Yields:
            str: The file path of each created video chunk.

        Raises:
            ValueError: If the video file cannot be opened.
            Exception: If `ffmpeg_extract_subclip` fails for a segment.
        """
        logger.info(f"Starting video splitting for {video_path} with chunk_interval={chunk_interval} seconds.")

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")

        frames_to_skip = int(fps * frames_per_second)
        curr_frame=0

        def seconds_to_hhmmss(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        base64Frames = []
        timestamps = []
        num_chunks = 0

        THRESHOLD=250000

        while curr_frame < total_frames:
            logger.debug(f"Processing frame {curr_frame}/{total_frames}")
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

            # to base64 encoding
            is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 25])
            base64frame = base64.b64encode(buffer).decode('utf-8')
            base64Frames.append(base64frame)

            # timestamp
            timestamp = seconds_to_hhmmss(curr_frame / fps)
            timestamps.append(timestamp)

            num_chunks = int(curr_frame / fps / chunk_interval)

            if num_chunks < int((curr_frame + frames_to_skip) / fps / chunk_interval):
                logger.info(f"Yielding chunk {num_chunks} with {len(base64Frames)} frames.")
                yield (base64Frames, timestamps)
                base64Frames.clear()
                timestamps.clear()

            curr_frame += frames_to_skip

        if len(base64Frames) > 0:
            logger.info(f"Yielding final chunk {num_chunks} with {len(base64Frames)} frames.")
            yield (base64Frames, timestamps)

        logger.info("Finished")
        video.release()
    
        return 


    def _analyze_chunk_frames(self, model: str, base64Frames: list, timestamps: list, prompt: str, **kwargs) -> str:
        """
        Analyzes a single chunk of video frames and returns a description.

        Args:
            base64Frames (list): The list of base64-encoded frames.
            timestamps (list): The list of timestamps corresponding to each frame.
            prompt (str): The prompt to guide the analysis.

        Returns:
            str: A description of the analyzed chunk.
        """
        logger.info(f"Analyzing chunk with {len(base64Frames)} frames.")
        #logger.info(f'Timestamps: {timestamps}')

        system_prompt =f"{self._chunk_description_prompt} \n\n# goal \n\n {prompt}"
        user_prompt = "These are the timestamps and the frames from the video."

        #logging.debug(f"system_prompt: {system_prompt}")
        #logging.debug(f"user_prompt: {user_prompt}")    

        try:

            user_content_parts = []
            # Add timestamp and image for each frame
            for b64frame, timestamp in zip(base64Frames, timestamps):
                user_content_parts.append(
                    {
                        "type": "text",
                        "text": f"Timestamp: {timestamp}"
                    }
                )
                user_content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64frame}",
                            "detail": "auto"
                        }
                    }
                )

            # Construct the messages for the chat completion API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": user_content_parts}
            ]

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            self._input_tokens += response.usage.prompt_tokens
            self._output_tokens += response.usage.completion_tokens

            results = response.choices[0].message.content

        except openai.APIError as e:
            #Handle API error here, e.g. retry or log
            logger.error(f"OpenAI API returned an API Error: {e}")
            results = f"OpenAI API returned an API Error: {e}"
        except openai.APIConnectionError as e:
            #Handle connection error here
            logger.error(f"Failed to connect to OpenAI API: {e}")
            results = f"Failed to connect to OpenAI API: {e}"
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            logger.error(f"OpenAI API request exceeded rate limit: {e}")
            results = f"OpenAI API request exceeded rate limit: {e}"

        return results
    

    def _aggregate_chunk_descriptions(self, model: str, chunk_descriptions: list, prompt: str, **kwargs) -> str:
        """
        Aggregates multiple chunk descriptions into a final summary.

        Args:
            chunk_descriptions (list): List of descriptions for each chunk.

        Returns:
            str: A comprehensive summary of the entire video.
        """

        logger.info("Aggregating chunk descriptions into final summary.")
        
        system_prompt ="You are an expert on Video Analysis. You will be shown a series of images from a video. Describe what is happening in the video, including the objects, actions, and any other relevant details. Be as specific and detailed as possible."
        user_prompt = "Please summarize the content to meet the following prompt: \n\n {prompt}"
        chunk_descriptions = f"These are the timestamps and detailed descriptions of video segments. \n\n {chunk_descriptions}"

        #logging.debug(f"system_prompt: {system_prompt}")
        #logging.debug(f"user_prompt: {user_prompt}")

        try:


            # Construct the messages for the chat completion API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": chunk_descriptions},
            ]

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            self._input_tokens += response.usage.prompt_tokens
            self._output_tokens += response.usage.completion_tokens

            final_summary = response.choices[0].message.content

        except openai.APIError as e:
            logger.error(f"OpenAI API returned an API Error during aggregation: {e}")
            final_summary = f"OpenAI API returned an API Error during aggregation: {e}"
        except openai.APIConnectionError as e:
            logger.error(f"Failed to connect to OpenAI API during aggregation: {e}")
            final_summary = f"Failed to connect to OpenAI API during aggregation: {e}"
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API request exceeded rate limit during aggregation: {e}")
            final_summary = f"OpenAI API request exceeded rate limit during aggregation: {e}"

        return final_summary


    def analyze_video_chunk(self, model: str, video_path: str, prompt: str, **kwargs) -> str:
        """
        Sends a prompt to ChatGPT and generates a response.

        Args:
            prompt (str): The prompt to provide to the LLM.
            model (str): The model identifier to use for the LLM.
            **kwargs: Additional parameters to pass to the OpenAI ChatCompletion API
                      (e.g., temperature, max_tokens, top_p, frequency_penalty, presence_penalty).

        Returns:
            str: The response text from ChatGPT.

        Raises:
            openai.error.OpenAIError: If an error occurs during the OpenAI API call.
        """
        if not self._client:
            raise ValueError("OpenAI client is not initialized.")
        logger.info(f"Generating response using model: {model}")
        analysis_start_time = time.time()
        all_chunk_descriptions = []
        current_time_offset_seconds = 0.0
        aggregated_summary = ""

        try:
            logger.info(f"Starting video analysis for: {video_path}")

            # Split video into chunks and process them
            for (base64Frames, timestamps) in self._split_video_into_chunks(video_path):
                chunk_analysis_start_time = time.time()
                response = self._analyze_chunk_frames(model, base64Frames, timestamps, prompt, **kwargs)
                logger.info(f"Chunk description: {response}")
                all_chunk_descriptions.append(response)
                chunk_analysis_end_time = time.time()
                logger.info(f"Chunk analysis time: {(chunk_analysis_end_time - chunk_analysis_start_time):.3f} seconds.")

            # Aggregate all chunk descriptions into a final summary
            logger.info("Aggregating chunk descriptions into final summary.")
            start_time_agg = time.time()
            aggregated_summary  = self._aggregate_chunk_descriptions(model, all_chunk_descriptions, prompt)
            end_time_agg = time.time()
            logger.info(f"summary: {aggregated_summary} ")
            logger.info(f"Aggregation time: {(end_time_agg - start_time_agg):.3f} seconds.")

        except Exception as e:
            logger.critical(f"An unhandled error occurred during video analysis: {e}", exc_info=True)
            return f"An error occurred during video analysis: {e}"
        finally:
            analysis_end_time = time.time()
            logger.info(f"Total video analysis execution time: {(analysis_end_time - analysis_start_time):.3f} seconds.")

        return aggregated_summary



class GeminiAccessor(LLMAccessor):
    """
    Provides access to Google Gemini (Generative AI).
    """
    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key, base_url)
        #if api_key is None:
        #    api_key = os.getenv("GEMINI_API_KEY")
        #if base_url is None:
        #    base_url = os.getenv("GEMINI_BASE_URL")
        if api_key is None :
            raise ValueError("Gemini API key and base URL must be set.")

        # Configure Gemini client
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai.types.HttpOptions(
                base_url=base_url
            )
        )

        logger.info("Initialized GeminiAccessor with base URL: %s", base_url)


    def process_vision(self, image_paths: List[str] = [], video_paths: List[str] = [] ) -> dict:
        """
        Processes visual data (images and videos) and prepares it for LLM input.

        Args:
            image_paths (List[str]): List of file paths to images.
            video_paths (List[str]): List of file paths to videos.
        Returns:
            dict: A dictionary containing processed visual data ready for LLM input.

        """
        logging.info("Processing visual data for LLM input.")
        results = []

        for image_path in image_paths:
            image = Image.open(image_path)

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

                results.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": image_base64
                    }
                })  

        # timestamp helper
        def seconds_to_hhmmss(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"


        for video_path in video_paths:
            base64Frames, seconds_per_frame = process_video(video_path)

            for i, frame_base64 in enumerate(base64Frames):
                timestamp = seconds_to_hhmmss(i * seconds_per_frame)
                results.append({
                    "text": f"timestamp: {timestamp}"
                    })
                results.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": frame_base64
                    }
                })

        return results
    

    def generate_response(self, model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Sends a prompt to Gemini and generates a response.

        Args:
            model (str): The model identifier to use for the LLM.
            system_prompt (str): The system prompt to provide to the LLM.
            user_prompt (str): The user prompt to provide to the LLM.
            **kwargs: Additional LLM-specific parameters (e.g., temperature, max_tokens).

        Returns:
            str: The response text from Gemini.

        """
        if not self._client:
            raise ValueError("Gemini client is not initialized.")
        logger.info(f"Generating response using model: {model}")


        try:
            # remove key "type"
            _user_prompt = copy.deepcopy(user_prompt)
            _system_prompt = copy.deepcopy(system_prompt)
            if isinstance(_user_prompt, list):
                for item in _user_prompt:
                    if isinstance(item, dict) and "type" in item.keys():
                        del item["type"]
            if isinstance(_system_prompt, list):
                for item in _system_prompt:
                    if isinstance(item, dict) and "type" in item.keys():
                        del item["type"]

            response = self._client.models.count_tokens(
                model=model,
                contents=_user_prompt,
            )
            logging.info(f"Token count: {response.total_tokens}")

            response = self._client.models.generate_content(
                model=model,
                contents=_user_prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                    system_instruction=_system_prompt
                ),
            )
            logger.info(f"Prompt tokens:\t {response.usage_metadata.prompt_token_count}")
            logger.info(f"Output tokens:\t {response.usage_metadata.candidates_token_count}")
            logger.info(f"--------------")
            logger.info(f"Total tokens:\t {response.usage_metadata.total_token_count}")

        except genai.errors.ClientError as e:
            print(f"Gemini API error occurred: {e}")
            raise # Re-raise the error to allow the caller to handle it
        
        return response.text



class OllamaAccessor(LLMAccessor):
    """
    Provides access to Ollama (OpenAI compatible).
    """
    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key, base_url)

        # Configure OpenAI client
        self._client = openai.OpenAI(
            api_key='ollama',
            base_url=base_url
        )

        logger.info("Initialized OllamaAccessor with base URL: %s", base_url)

    def process_vision(self, image_paths: List[str] = [], video_paths: List[str] = [] ) -> List[dict]:
        """
        Processes visual data (images and videos) and prepares it for LLM input.

        Args:
            image_paths (List[str]): List of file paths to images.
            video_paths (List[str]): List of file paths to videos.
        Returns:
            dict: A dictionary containing processed visual data ready for LLM input.

        """
        logging.info("Processing visual data for LLM input.")
        results = []

        for image_path in image_paths:
            image = Image.open(image_path)

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

                results.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "low"
                    }
                })  

        # timestamp helper
        def seconds_to_hhmmss(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"


        for video_path in video_paths:
            base64Frames, seconds_per_frame = process_video(video_path)

            for i, frame_base64 in enumerate(base64Frames):
                timestamp = seconds_to_hhmmss(i * seconds_per_frame)
                results.append({
                    "type": "text",
                    "text": f"timestamp: {timestamp}"
                    })
                results.append({
                    "type": "image_url",
                    "image_url": {
                        'url': f"data:image/jpeg;base64,{frame_base64}",
                        'detail': 'low'
                    }
                })

        return results


    def generate_response(self, model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Sends a prompt to ChatGPT and generates a response.

        Args:
            system_prompt (str): The system prompt to provide to the LLM.
            user_prompt (str): The user prompt to provide to the LLM.
            model (str): The model identifier to use for the LLM.
            **kwargs: Additional parameters to pass to the OpenAI ChatCompletion API
                      (e.g., temperature, max_tokens, top_p, frequency_penalty, presence_penalty).

        Returns:
            str: The response text from ChatGPT.

        Raises:
            openai.error.OpenAIError: If an error occurs during the OpenAI API call.
        """
        if not self._client:
            raise ValueError("OpenAI client is not initialized.")
        
        if isinstance(system_prompt[0], str):
            system_prompt = [{'type': 'text', 'text': system_prompt[0]}]

        logger.info(f"system_prompt: {system_prompt}")
        logger.info(f"Generating response using model: {model}")
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ollama API error occurred: {e}")
            raise # Re-raise the error to allow the caller to handle it

