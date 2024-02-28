import os


class Config:
    repository = "http://localhost:7200/repositories/KIDZ"
    repository_update = repository + "/statements"
    openai_api_key = os.getenv("gptkidz")

    mongodb_client = 'mongodb://localhost:27017/'
    mongodb_database = 'datalake'
