import os

import openai
import pandas as pd

from config import Config

openai.api_key = Config.openai_api_key
df = pd.read_csv("out_openai_completion.csv")

prepared_data = df.loc[:, ['sub_prompt', 'response_txt']]
prepared_data.rename(columns={'sub_prompt': 'prompt', 'response_txt': 'completion'}, inplace=True)
prepared_data.to_csv('prepared_data.csv', index=False)

subprocess.run('openai tools fine_tunes.prepare_data --file prepared_data.csv --quiet'.split())

## Start fine-tuning
subprocess.run(
    'openai api fine_tunes.create --training_file prepared_data_prepared.jsonl --model davinci --suffix "SuperHero"'.split())
