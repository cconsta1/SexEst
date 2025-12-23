# Contributing to SexEst

Thank you for your interest in contributing. This file summarizes how to report issues, suggest documentation improvements, and reproduce model training.

Reporting issues
- Open an issue describing the problem, expected behavior, and steps to reproduce. Include system details and a minimal example input where possible.

Documentation and small fixes
- PRs that improve documentation, README clarity, or example inputs are welcome. Keep changes focused and include rationale in the PR description.

About the models
- The repository contains pre-trained models and associated metadata in the `models_*` folders. These are the canonical models used by the live demo. Do not overwrite or remove these files unless you are intentionally retraining and versioning new models.
- Model training notebooks are available at: https://github.com/cconsta1/SexEst_Notebooks.git

Reproducing locally
1. Create and activate a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run the app locally:
```bash
streamlit run streamlit_app.py
```

Data provenance and citation
- The training data used (Goldman osteometric and Howells craniometric datasets) are available from Dr. B. Auerbach: https://web.utk.edu/~auerbach/DATA.htm. Please follow the dataset owners' citation and usage guidance when using these data.

License and citation
- This project is distributed under the Apache License 2.0. If you use SexEst in research, please cite the associated paper: https://doi.org/10.1002/oa.3109

Contributor etiquette
- Keep PRs small and focused.
- Include tests or reproducible examples for functional changes where feasible.
- Respect the existing license and dataset citation requirements.
