import sys
import os

# Ensure project src dir is importable
BASE_DIR = os.path.normpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Streamlit entrypoint: importing the dashboard module executes the app
from dashboard import trayectorias_dashboard  # noqa: F401

if __name__ == '__main__':
    print('Run this app with:')
    print('  streamlit run app.py')
