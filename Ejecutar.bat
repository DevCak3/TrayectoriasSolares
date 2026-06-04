@echo off
rem Configura la ruta a Miniforge/Conda y el nombre del entorno aquí
set "CONDA_PATH=C:\CODE\miniforge"
set "ENV_NAME=gis"

if not exist "%CONDA_PATH%\Scripts\activate.bat" (
	echo No se encontro el script de activacion en %CONDA_PATH%\Scripts\activate.bat
	echo Ajusta la variable CONDA_PATH en este archivo.
	pause
	exit /b 1
)

call "%CONDA_PATH%\Scripts\activate.bat" "%ENV_NAME%"
if errorlevel 1 (
	echo Error al activar el entorno %ENV_NAME%
	pause
	exit /b 1
)

rem Ejecuta Streamlit usando la ruta relativa al .bat
streamlit run "%~dp0src\dashboard\dashboard.py"
pause