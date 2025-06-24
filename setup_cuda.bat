@echo off
echo ==========================================
echo Slime Wallpaper CUDA Setup Helper
echo ==========================================
echo.
echo This script will help you set up CUDA for GPU acceleration.
echo.

:: Check if CUDA is already installed
where nvcc >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo CUDA Toolkit is already installed.
    nvcc --version
    goto :cuda_vars
) else (
    echo CUDA Toolkit is not found in your PATH.
    echo.
    echo Please download and install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    echo After installation, run this script again.
    goto :options
)

:cuda_vars
echo.
echo Checking environment variables...

:: Check if CUDA_PATH is set
if defined CUDA_PATH (
    echo CUDA_PATH is set to: %CUDA_PATH%
) else (
    echo CUDA_PATH is not set!
    echo.
    
    :: Try to find CUDA installation directory
    for /D %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*.*") do (
        set "CUDA_FOUND=%%i"
    )
    
    if defined CUDA_FOUND (
        echo Found CUDA installation at: %CUDA_FOUND%
        echo.
        echo To set CUDA_PATH, run:
        echo setx CUDA_PATH "%CUDA_FOUND%"
    ) else (
        echo Could not find CUDA installation directory.
    )
)

:: Check if NVVM DLL is accessible
if exist "%CUDA_PATH%\nvvm\bin\nvvm64_40_0.dll" (
    echo NVVM DLL found at: %CUDA_PATH%\nvvm\bin\nvvm64_40_0.dll
) else (
    echo NVVM DLL not found! This is required for GPU acceleration.
)

echo.
echo To ensure all CUDA libraries are in your PATH, add these directories:
echo %CUDA_PATH%\bin
echo %CUDA_PATH%\nvvm\bin
echo.
echo To add them, run:
echo setx PATH "%%PATH%%;%CUDA_PATH%\bin;%CUDA_PATH%\nvvm\bin"

:options
echo.
echo Options:
echo 1. Install Python dependencies
echo 2. Exit
echo.
set /p choice=Enter your choice (1-2): 

if "%choice%"=="1" (
    echo Installing Python dependencies...
    pip install -r requirements.txt
    echo.
    echo For GPU acceleration, consider using Anaconda:
    echo conda install numba cudatoolkit=11.8
)

echo.
echo Setup completed. Press any key to exit.
pause >nul
