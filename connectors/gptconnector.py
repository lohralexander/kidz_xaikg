import time

from config import Config, logger


def gpt_request(user_message, system_message=None, previous_conversation=None, retrieved_information=None,
                model="gpt-4o-2024-11-20",
                sleep_time=0, seed=42, temperature=0):
    client = Config.chatgpt_client
    message = []
    if previous_conversation is not None:
        message = previous_conversation
        message.append({"role": "user", "content": user_message})

    if system_message is not None:
        message.append({"role": "system", "content": system_message})

    if retrieved_information is not None:
        logger.debug(f"USED RETRIEVED INFORMATION FOR REQUEST: {retrieved_information}")
        message.append({"role": "user",
                        "content": f"{user_message}. Here is ontology retrieved information based on the topic: {retrieved_information}"})
    else:
        message.append({"role": "user", "content": user_message})
    logger.debug(f"Message: {message}")
    try:
        response = client.chat.completions.create(
            temperature=temperature,
            messages=message,
            model=model,
            seed=seed
        )
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None, previous_conversation
    if response and response.choices:
        response_message = response.choices[0].message.content
        message.append({"role": "assistant", "content": response_message})
        conversation = message
    else:
        response_message = "Error: No valid response from API."
        conversation = message

    logger.debug(f"Assistant: {response_message}")
    time.sleep(sleep_time)
    return response_message, conversation
