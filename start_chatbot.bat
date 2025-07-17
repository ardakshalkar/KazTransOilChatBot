@echo off
echo Starting KazTransOil Knowledge Chatbot...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Start Streamlit app
echo Virtual environment activated!
echo Starting Streamlit application...
echo.
echo The chatbot will be available at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

echo.
echo Chatbot application stopped.
pause 