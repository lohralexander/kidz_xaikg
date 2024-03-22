@echo off
setlocal enabledelayedexpansion
SETLOCAL ENABLEEXTENSIONS

set containerName=weaviate
set count=0

for /f %%i in ('docker ps -q -a --filter "name=%containerName%"') do (
  set /a count+=1
)

if %count% gtr 0 (
  echo %containerName% container already exists. Starting existing container.
  docker start %containerName%
) else (
  echo %containerName% container does not exist. Pulling image and starting container.
  docker run -d --name weaviate --restart on-failure:0 -p 8080:8080 -p 50051:50051 -v weaviate_data:/var/lib/weaviate -e QUERY_DEFAULTS_LIMIT=25 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -e DEFAULT_VECTORIZER_MODULE=none -e ENABLE_MODULES=text2vec-cohere,text2vec-huggingface,text2vec-palm,text2vec-openai,generative-openai,generative-cohere,generative-palm,ref2vec-centroid,reranker-cohere,qna-openai -e CLUSTER_HOSTNAME=node1 semitechnologies/weaviate:1.23.10 --host 0.0.0.0 --port 8080 --scheme http
)