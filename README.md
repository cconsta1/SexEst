# SexEst

Live demo: http://sexest.cyi.ac.cy/  
Paper / DOI: https://doi.org/10.1002/oa.3109

Short description
- SexEst is an open-source Streamlit web application for predicting biological sex from skeletal measurements using pre-trained machine learning models (XGBoost, LightGBM, Linear Discriminant Analysis).

Background
- Skeletal sex estimation is an essential step in osteoarchaeological and forensic contexts. This project (1) evaluates multiple machine-learning classifiers on worldwide cranial and postcranial measurements and (2) deploys the best-performing models in a free web application for straightforward sex prediction of unknown skeletons. Selected text from the paper: “Skeletal sex estimation is an essential step in any osteoarcheological study... The models offering the highest rates of correct sex classification (Extreme Gradient Boosting, Light Gradient Boosting, and Linear Discriminant Analysis) were then selected to construct an open access and open source web application, SexEst.”

Key links
- Live app: http://sexest.cyi.ac.cy/
- Paper / DOI: https://doi.org/10.1002/oa.3109
- Model training notebooks: https://github.com/cconsta1/SexEst_Notebooks.git
- Original datasets (Goldman & Howells): https://web.utk.edu/~auerbach/DATA.htm

What is in this repository
- `streamlit_app.py` — Streamlit web UI and inference logic (loads pre-trained models and shows predictions).
- `models_goldman/`, `models_howell/` — pre-trained model metadata (and in some cases model files).
- `sample_dataset_craniometric.csv`, `sample_dataset_osteometric.csv` — example input files.
- `requirements.txt` — Python dependencies required to run the app locally.
- `LICENSE` — Apache License 2.0 (this repository is distributed under Apache 2.0).

Quickstart — run locally
1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
3. Open http://localhost:8501/ in your browser.

Docker (optional)
- A `Dockerfile` is included for building a containerized version of the app. To build and run the image:
```bash
docker build -f Dockerfile -t app:latest .
docker run -p 8501:8501 app:latest
```
Visit http://localhost:8501/ (or the mapped host port) once the container is running.

Notes on models and data
- The app uses pre-trained models; training notebooks used to produce those models are available at https://github.com/cconsta1/SexEst_Notebooks.git. The original training datasets (Goldman osteometric and Howells craniometric) are freely available from Dr. B. Auerbach: https://web.utk.edu/~auerbach/DATA.htm — please follow the dataset owners' citation guidelines if you reuse the data.
- Please do not modify or replace the packaged models in `models_*` unless you intend to retrain and version them appropriately.

Contributing
- See `CONTRIBUTING.md` for guidance on reporting issues, documentation edits, and reproducing the analysis.

Recommended housekeeping
- Add a `.gitignore` to avoid committing virtual environments, caches, or large model binaries.
- Consider adding badges (license, demo link) and an explicit `DATA_AVAILABILITY.md` to document the provenance and citation of the datasets used.

License
- This repository is licensed under the Apache License 2.0. See `LICENSE` for details.

Contact & citation
- If you use SexEst for research, please cite the associated paper: https://doi.org/10.1002/oa.3109
