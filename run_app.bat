@echo off
cd /d "%~dp0"
call venv\Scripts\activate
start "RAG Doc QA" cmd /k streamlit run app.py
timeout /t 6 /nobreak >nul
start http://localhost:8501
