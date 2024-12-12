import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))


def llm_fuzzy_match(pred: str, reference: str, question: str, disallow_partial: bool=True) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()

    print(response)

    if "incorrect" in response or ("partially correct" in response and disallow_partial):
        return 0.0
    else:
        assert "correct" in response, response
        return 1.0

def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    answer: str = response.choices[0].message.content
    return answer