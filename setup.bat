@echo off
setlocal

REM Set project variables
set VENV_DIR=venv
set APP_FILE=app.py

REM Step 1: Create virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Step 2: Activate virtual environment
call %VENV_DIR%\Scripts\activate

REM Step 3: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Step 4: Install required dependencies
echo Installing dependencies...
pip install flask transformers pyspellchecker rapidfuzz pandas numpy scikit-learn torch

REM Step 5: Run the app
echo Running the Flask app...
python %APP_FILE%

pause
