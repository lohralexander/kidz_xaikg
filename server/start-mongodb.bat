@echo off
setlocal enabledelayedexpansion

set containerName=mongodb
set count=0

for /f %%i in ('docker ps -q -a --filter "name=%containerName%"') do (
  set /a count+=1
)

if %count% gtr 0 (
  echo %containerName% container already exists. Starting existing container.
  docker start %containerName%
) else (
  echo %containerName% container does not exist. Pulling image and starting container.
  docker run --name mongodb -p 27017:27017 mongo:7.0.3
)