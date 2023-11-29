@echo off
setlocal enabledelayedexpansion

set containerName=graphdb
set count=0

for /f %%i in ('docker ps -q -a --filter "name=%containerName%"') do (
  set /a count+=1
)

if %count% gtr 0 (
  echo %containerName% container already exists. Starting existing container.
  docker start %containerName%
) else (
  echo %containerName% container does not exist. Pulling image and starting container.
  docker run --name %containerName% -p 7200:7200 -p 7300:7300 alexanderlohr91/kidz:latest
)