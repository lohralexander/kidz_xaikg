import logging
import os
from datetime import datetime


class Config:

    graph_gen = True

    graphdb_repository = "http://localhost:7200/repositories/KIDZ"
    graphddb_repository_update = graphdb_repository + "/statements"
    graphdb_connector = "http://localhost:7200/rest/chat/retrieval"

    openai_api_key = os.getenv("gptkidz")

    mongodb_client = 'mongodb://localhost:27017/'
    mongodb_database = 'datalake'

    chatgpt_retrieval_plugin = "localhost:8100"
    # on my machine, the retrieval plugin is running on port 8100, because 8000 caused conflicts with other services
    # on your machine, you might want to use 8000, which is the default port for the retrieval plugin


class Logger:
    @staticmethod
    def setup_logging():
        # Create logger
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)

        # Create file handler
        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.basicConfig(filename=f'research_log_{datetime.now().strftime("%Y_%m_%d_%H_%M")}.txt',
                            level=logging.INFO)

        return logger


logger = Logger.setup_logging()
