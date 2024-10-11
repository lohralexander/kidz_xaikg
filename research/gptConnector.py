import time

from openai import OpenAI

from config import Config
from config import logger


def gpt_request(system, user, model="gpt-4o", sleep_time=5):
    client = OpenAI(
        api_key=Config.openai_api_key,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"{system}"
            },
            {
                "role": "user",
                "content": f"{user}"
            }
        ],
        model="gpt-4o",
    )
    response_message = response.choices[0].message.content
    logger.info(f"System: {system}")
    logger.info(f"User: {user}")
    logger.info(f"Assistant: {response_message}")
    time.sleep(sleep_time)
    return response_message
