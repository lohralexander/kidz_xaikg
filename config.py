import os


class Config:
    graphdb_repository = "http://localhost:7200/repositories/KIDZ"
    graphddb_repository_update = graphdb_repository + "/statements"
    graphdb_connector = "http://localhost:7200/rest/chat/retrieval"

    openai_api_key = os.getenv("gptkidz")

    mongodb_client = 'mongodb://localhost:27017/'
    mongodb_database = 'datalake'

    chatgpt_retrieval_plugin = "localhost:8100"
    # on my machine, the retrieval plugin is running on port 8100, because 8000 caused conflicts with other services
    # on your machine, you might want to use 8000, which is the default port for the retrieval plugin

