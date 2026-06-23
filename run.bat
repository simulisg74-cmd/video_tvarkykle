@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Kuriama virtuali aplinka...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo Paleidziama Streamlit aplikacija...
.venv\Scripts\python.exe -m streamlit run app.py
