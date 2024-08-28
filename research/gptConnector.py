import openai
import time
from config import Config


def gpt_request(system, user, model="gpt-4o", sleep_time=20):
    openai.api_key = Config.openai_api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{system}"},
            {"role": "user", "content": f"{user}"}
        ]
    )
    time.sleep(sleep_time)
    return response.choices[0].message['content']
