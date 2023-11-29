for /f %%i in ('docker ps -qf "name=mongodb"') do set containerId=%%i
echo %containerId%
If "%containerId%" == "" (
  echo "No Container running"
) ELSE (
  docker stop %ContainerId%
)

for /f %%i in ('docker ps -qf "name=graphdb"') do set containerId=%%i
echo %containerId%
If "%containerId%" == "" (
  echo "No Container running"
) ELSE (
  docker stop %ContainerId%
)