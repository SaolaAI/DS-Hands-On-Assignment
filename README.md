# Project Overview
Machine learning pipeline with FastAPI backend and Streamlit frontend interface.

## Setup
```bash
pip install -r requirements.txt
```

## Project Structure
- `ui.py`: FastAPI backend serving machine learning models
- `streamlit_app.py`: Streamlit frontend for user interaction
- `models.py`: Core machine learning model implementations
- `*.pkl`: Serialized model artifacts and preprocessing pipelines
- `train_data.json`/`val_data.json`: Training and validation datasets

## Usage
1. Start FastAPI backend:
```bash
uvicorn ui:app --reload
```
2. Start Streamlit frontend in a new terminal:
```bash
streamlit run streamlit_app.py
```

Access the Streamlit interface at http://localhost:8501 and explore API endpoints at http://localhost:8000/docs
