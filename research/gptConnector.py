import json
import time

from openai import OpenAI

from config import Config
from config import logger


def gpt_request(system, user, model="gpt-4o", sleep_time=5):
    client = OpenAI(
        api_key=Config.openai_api_key,
    )

    messages = [
        {
            "role": "system",
            "content": f"{system}"
        },
        {
            "role": "user",
            "content": f"{user}"
        }
    ]

    response = client.chat.completions.create(seed=42,
                                              temperature=0,
                                              messages=messages,
                                              model="gpt-4o",

                                              )
    response_message = response.choices[0].message.content
    logger.info(f"System: {system}")
    logger.info(f"User: {user}")
    logger.info(f"Assistant: {response_message}")
    time.sleep(sleep_time)
    return response_message


def gpt_request_new(message, previous_messages=None, model="gpt-4o", sleep_time=5):
    client = OpenAI(
        api_key=Config.openai_api_key,
    )

    if previous_messages is not None:
        message = previous_messages+message
    logger.info(f"User: {message}")
    response = client.chat.completions.create(seed=42,
                                              temperature=0,
                                              messages=message,
                                              model="gpt-4o",
                                              )
    response_message = response.choices[0].message.content
    conversation = message + [{f"role": "assistant", "content": f"{response_message}"}]

    logger.info(f"Assistant: {response_message}")
    time.sleep(sleep_time)
    return response_message, conversation
