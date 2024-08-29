import time

from openai import OpenAI

from config import Config


def gpt_request(system, user, model="gpt-4o", sleep_time=20):
    # openai.api_key = Config.openai_api_key
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": f"{system}"},
    #         {"role": "user", "content": f"{user}"}
    #     ]
    # )

    client = OpenAI(
        # This is the default and can be omitted
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
    time.sleep(sleep_time)
    return response.choices[0].message.content
