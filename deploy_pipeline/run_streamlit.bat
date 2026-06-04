@echo off
REM Activate conda environment 'gis' and run streamlit app
call "c:\CODE\miniforge\shell\condabin\conda-hook.ps1"
conda activate gis
cd /d "%~dp0"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
