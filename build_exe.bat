@echo off
echo Building MeshIt executable...

REM Try to find the correct Python environment
set PYTHON_EXE=C:\Users\Waqas Hussain\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo Error: Python environment not found at %PYTHON_EXE%
    echo Looking for alternative Python installations...
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('where python') do set PYTHON_EXE=%%i
        echo Using Python from PATH: %PYTHON_EXE%
    ) else (
        echo No Python installation found. Please install Python and PyInstaller.
        pause
        exit /b 1
    )
)

echo Using Python: %PYTHON_EXE%

REM Run PyInstaller with the correct Python environment
"%PYTHON_EXE%" -m pyinstaller ^
    --name=MeshIt ^
    --windowed ^
    --onefile ^
    --noconfirm ^
    main.py ^
    --add-data=resources;resources ^
    --add-data=Pymeshit;Pymeshit ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=shiboken6 ^
    --hidden-import=scipy ^
    --hidden-import=scipy.sparse ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.pyplot ^
    --hidden-import=pyvista ^
    --hidden-import=tetgen ^
    --hidden-import=triangle ^
    --icon=resources\images\app_logo_small.png ^
    --distpath=release ^
    --workpath=build

if %errorlevel% equ 0 (
    echo.
    echo ✅ Build completed successfully!
    echo You can find the executable in the 'release' folder
    dir release\*.exe
) else (
    echo.
    echo ❌ Build failed!
)

pause
