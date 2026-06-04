Deployment pipeline for the Trayectorias Solar dashboard

Contents:
- `app.py`: Streamlit entrypoint that imports the dashboard module from `src/`.
- `requirements.txt`: Python packages needed for deployment.
- `fetch_tmy.py`: helper script to download a sample NASA POWER hourly file into `data/`.
- `run_streamlit.bat`: Windows batch that activates the `gis` conda env and runs Streamlit.

Quick local run (Windows, conda):

1. From project root, create a virtual env or use conda environment `gis`.
2. Install requirements:

```powershell
conda activate gis
pip install -r deploy_pipeline/requirements.txt
```

3. (Optional) Fetch TMY data:

```powershell
python deploy_pipeline/fetch_tmy.py
```

4. Run the app:

```powershell
cd deploy_pipeline
streamlit run app.py
```

Deploying to Hugging Face Spaces / Streamlit Cloud:

- Push the repository to GitHub.
- On Hugging Face: create a new Space (type Streamlit) and connect the repo branch. Ensure `requirements.txt` is at the repo root or in the Space settings.
- On Streamlit Cloud: create a new app pointing to the GitHub repo and the path `deploy_pipeline/app.py`.

Secrets / API keys:
- If you add APIs that need keys, use the hosting provider's Secrets feature. Do NOT commit keys to the repo.
