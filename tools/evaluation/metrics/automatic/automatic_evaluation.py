import os
from openai import OpenAI
from typing import Dict, List, Tuple, Any

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
import json
import re

from FWA_evaluator_logger import set_logger, getLogger
set_logger()
logger = getLogger(__name__)


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL")
)


MODEL_LLM_AS_A_JUDGE = "gpt-4o-2024-08-06"
SEED=42

def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> Tuple[str, int, int]:
    ''' Generate text from OpenAI chat completion API.
    Args:
        messages (list[dict[str, str]]): The messages to send to the model.
        model (str): The model to use.
        temperature (float): The temperature to use.
        max_tokens (int): The maximum number of tokens to generate.
        seed (int): The seed to use for random generation.
        top_p (float): The top_p to use.
        context_length (int): The context length to use.
        stop_token (str | None): The stop token to use.
    Returns:
        Tuple[str, int, int]: The generated text, input tokens, and output tokens.
    '''
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    
    # using stop token
    stop_param = [stop_token] if stop_token else None

    try:

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
            stop=stop_param,
        )

        answer: str = response.choices[0].message.content
        input_tokens: int = response.usage.prompt_tokens
        output_tokens: int = response.usage.completion_tokens

        return answer, input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        return "", 0, 0



def llm_fuzzy_match(pred: str, reference: str, question: str) -> Tuple[float, None, int, int]:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    #message += "Also answer the reason why you judged so."

    logger.info(f"message: {message}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    try:
        response, input_tokens, output_tokens = generate_from_openai_chat_completion(
            #model="gpt-4-1106-preview",
            model=MODEL_LLM_AS_A_JUDGE,
            messages=messages,
            temperature=0,
            max_tokens=768,
            top_p=1.0,
            context_length=0,
            seed=SEED,
        )
        response = response.lower()

        logger.info(f"response: {response}")
        if "partially correct" in response or "incorrect" in response:
            return 0.0, None, input_tokens, output_tokens
        else:
            assert "correct" in response, response
            return 1.0, None, input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error in LLM fuzzy match: {e}")
        return 0.0, None, 0, 0


def clean_answer(answer: str) -> str:
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower(), None


#def exact_match(ref: str, pred: Union[str, int]) -> float:
def exact_match(ref: str, pred: str) -> float:
    if isinstance(pred, int):
        pred = str(pred)
    return float(
        clean_answer(pred) == clean_answer(ref)
    ), None


def must_include(ref: str, pred: str) -> float:
    clean_ref = clean_answer(ref)
    clean_pred = clean_answer(pred)
    # tokenize the answer if the ref is a single word
    # prevent false positive (e.g, 0)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref in tok_pred), None
    else:
        return float(clean_ref in clean_pred), None


def must_exclude(ref: str, pred: str) -> float:
    """Returns 1 if pred is not in ref, and 0 otherwise"""
    clean_ref = clean_answer(ref)
    clean_pred = clean_answer(pred)
    # tokenize the answer if the ref is a single word
    # prevent false positive (e.g, 0)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref not in tok_pred), None
    else:
        return float(clean_ref not in clean_pred), None


def json_match(pred: str, reference: str, question: str) -> Tuple[float, str, int, int]:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    message += "Answer is given in JSON format. so you should compare the number of incidents, violations or other things and the keys of the answer"
    message += "Also answer the reason why the answer is correct, incorrect or partially correct"
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    try:
        response, input_tokens, output_tokens = generate_from_openai_chat_completion(
            model=MODEL_LLM_AS_A_JUDGE,
            messages=messages,
            temperature=0,
            max_tokens=768,
            top_p=1.0,
            context_length=0,
            seed=SEED,
        )

        response = response.lower()

        if reference == "[ ]":
            return 0.0, None, input_tokens, output_tokens

        # #"=================================")
        # import json
        # print("question: ", question)
        # print("pred: ", pred)
        # print("reference: ", reference)

        # print("response: ", response)
        if "partially correct" in response or "incorrect" in response:
            return 0.0, response.replace("\n", " "), input_tokens, output_tokens
        else:
            assert "correct" in response, response
            return 1.0, response.replace("\n", " "), input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error in JSON match: {e}")
        return 0.0, None, 0, 0

def eval_distance(pred: float, reference: float) -> float:
    ratio_threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
    score_candidates = [1.0, 0.8, 0.6, 0.4, 0.2]
    if reference == 0:
        raise ValueError("Reference value cannot be zero for distance evaluation.")
    difference = abs(pred - reference)
    ratio = difference / reference
    for threshold in ratio_threshold:
        if ratio <= threshold:
            return score_candidates[ratio_threshold.index(threshold)]
    return 0.0


def eval_time(pred: float, reference: float) -> float:
    # should depend on the length of the video
    diff_threshold = [1, 5, 10, 30, 60]
    score_candidates = [1.0, 0.8, 0.6, 0.4, 0.2]

    difference = abs(pred - reference)
    for threshold in diff_threshold:
        if difference <= threshold:
            return score_candidates[diff_threshold.index(threshold)]
    return 0.0

def numerical_match(pred: str, reference:str, question: str, numerical_ratio = 0.5) -> Tuple[float, str, int, int]:

    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question.\n"
    message = "The teacher evaluate numerical values by themselves, so you must retrieve the numerical values (e.g : number of objects, time, length) and give it to the teacher.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Your task consist of two steps:\n"
    message += "1. Compare the non-numerical part of the answer and determine if the answer is correct, incorrect or partially correct.\n"
    message += "2. Extract the numerical values from the question and the answers.\n"

    message += "Give the numerical values asked in the question from the both answers separately.\n"
    message += "If the answer does not contain any numerical values or only contains relative values (e.g., more, less, higher, lower, \"<\", \">\"), you should answer 'N/A' for the numerical values."
    message += """
You MUST ANSWER JSON FORMAT BELOW:
{
    "correctness": "correct/incorrect/partially correct",
    "numerical_values":
    {"KEY1": {"teacher": "VALUE_T1", "student": "VALUE_S1", "unit": "UNIT1", "type": "TYPE1"},
     "KEY2": {"teacher": "VALUE_T2", "student": "VALUE_S2", "unit": "UNIT2", "type": "TYPE2"},    
    ...
    }
}
...
KEY; The key of the numerical value extracted from the question. (e.g., "number of *objects*", "time", "distance") 
UNIT: The unit of the numerical value. (e.g., "m", "cm", "s", "min" or name of the object)
TYPE: The type of the numerical value. ("length", "time", "number")
All values should be numerical values. If the units are different, you should convert the units to the same unit. (SI unit is recommended).
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    try:
        response, input_tokens, output_tokens = generate_from_openai_chat_completion(
            model=MODEL_LLM_AS_A_JUDGE,
            messages=messages,
            temperature=0,
            max_tokens=768,
            top_p=1.0,
            context_length=0,
            seed=SEED,
        )
    except Exception as e:
        logger.error(f"Error in JSON match: {e}")
        return 0.0, None, 0, 0
    
    # Extract JSON from the response
    json_pattern = re.compile(r'\{.*\}', re.DOTALL)
    json_match = json_pattern.search(response)
    
    if not json_match:
        return 0.0, {}, input_tokens, output_tokens

    json_str = json_match.group(0)
    #print("json_str: ", json_str)  

    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError:
        print("response: ", response)

        print("JSON Decode Error", json_str)
        return 0.0, {}, input_tokens, output_tokens

    #print(json_data)
    # Further processing of json_data if needed
    
    if json_data["correctness"] == "incorrect":
        score = 0.0
    else:
        numerical_score = 0.0

        for _, v in json_data["numerical_values"].items():
            print("v: ", v)
            value_t = v["teacher"]
            value_s = v["student"]
            if 'type' in v:
                value_type = v["type"]
            else:
                if 'type' in value_t:
                    value_type = value_t['type']

            match value_type:
                case "number":
                    try:
                        value_t = int(value_t)
                        value_s = int(value_s)
                        if value_t == value_s:
                            numerical_score += 1.0                        
                    except:
                        pass

                case "time":
                    def convert_to_seconds(time_str: str) -> float:
                        """Convert time string in hh:mm:ss format to seconds."""
                        parts = time_str.split(':')
                        if len(parts) == 3:
                            hours, minutes, seconds = map(float, parts)
                            return hours * 3600 + minutes * 60 + seconds
                        elif len(parts) == 2:
                            minutes, seconds = map(float, parts)
                            return minutes * 60 + seconds
                        elif len(parts) == 1:
                            return float(parts[0])
                        else:
                            raise ValueError(f"Invalid time format: {time_str}")
                    try:
                        if type(value_t) == str:
                            value_t = convert_to_seconds(value_t)
                        if type(value_s) == str:
                            value_s = convert_to_seconds(value_s)
                        numerical_score += eval_time(value_s, value_t)
                    except:
                        pass
                case "length":
                    try:
                        value_t = float(value_t)
                        value_s = float(value_s)
                        numerical_score += eval_distance(value_s, value_t)
                    except:
                        pass    
                case _:
                    print("Unknown type: ", type)
                    pass

        numerical_score /= len(json_data["numerical_values"])
        #print("numerical_score: ", numerical_score)
        #print("response: \n", response)
        score = (1 - numerical_ratio) + numerical_score * numerical_ratio
    return score, json_data, input_tokens, output_tokens