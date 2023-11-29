class Config:
    # Path to GraphML repository
    repository = "http://localhost:7200/repositories/KIDZ"
    repository_update = repository + "/statements"
    model_name = 'test.pickle'

    mongodb_client = 'mongodb://localhost:27017/'
    mongodb_database = 'datalake'
