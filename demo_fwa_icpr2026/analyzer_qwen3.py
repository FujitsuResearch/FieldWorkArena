import os
import logging

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# Qwen/Qwen3-VL-2B-Instruct, "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", Qwen/Qwen3-VL-32B-Instruct
QWEN3_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# default: Load the model on the available device(s)
#model = AutoModelForImageTextToText.from_pretrained(
#    QWEN3_MODEL,
#    dtype=torch.bfloat16,
#    device_map="auto"
#)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    QWEN3_MODEL,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(QWEN3_MODEL)

def query_video_qwen3(video_path: str, prompt: str):
    logger.info(f"Processing video: {video_path}, prompt: {prompt}")

    # Create messages structure for the entire video
    total_pixels=24576  * 32 * 32
    min_pixels=4 * 32 * 32
    max_pixels=256 * 32 * 32

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{video_path}",
                    "min_pixels": min_pixels, 
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                    #"fps": 0.16666,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True, 
        image_patch_size= 16,
        return_video_metadata=True
    )
    
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        logger.info(f"video_inputs length: {video_inputs[0].shape}")
    else:
        video_metadatas = None

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Inference
    logger.info("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def query_image_qwen3(image_path: str, prompt: str):
    logger.info(f"Processing video: {image_path}, prompt: {prompt}")

    # Create messages structure for the entire video
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}"
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        logger.info(f"Videos: {videos[0].shape}")
    else:
        video_metadatas = None

    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
    inputs = inputs.to(model.device)

    # Inference
    logger.info("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


if __name__ == "__main__":
    result = query_video_qwen3(
        video_path="/data1/dataset/FieldWorkArena_v1/data/movie/West5_G210_HANKUMI_00h03m50s_00h09m02s.mp4",
        prompt="In this video, what are the start and end times of the elemental action \"quality check with two check masks\"?"
    )
    # "The specified element motion start and end times are from 00:03:18 to 00:03:24."
    print(result)
