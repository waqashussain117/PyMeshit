@echo off
echo Cleaning __pycache__ directories and build files...

REM Clean __pycache__ directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

REM Clean other cache files
if exist "*.pyc" del /q "*.pyc"

REM Clean build directories
if exist "build" rd /s /q "build"
if exist "release" rd /s /q "release"
if exist "MeshIt.spec" del /q "MeshIt.spec"

echo Cleanup completed!
pause
