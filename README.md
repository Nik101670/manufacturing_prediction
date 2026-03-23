# Manufacturing Prediction

🎯 Overview
A machine-learning-powered manufacturing prediction platform that serves model inference through an API and interactive Streamlit analytics dashboards.

🚀 Features
- Real-time manufacturing prediction via FastAPI endpoint (low-latency API inference)
- Interactive model performance and output analysis dashboards in Streamlit

🛠 Tech Stack
- Python
- FastAPI
- Streamlit
- Scikit-learn
- PyTorch (CPU)
- Pandas, NumPy, SciPy
- Plotly
- Uvicorn
- Docker, Docker Compose

📈 Results
- Model evaluation artifacts and metrics are available in the model directory.
- Dashboard pages provide performance and output analysis for the 1000-sample manufacturing dataset.

📋 Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the API backend:
   - `uvicorn app.api:app --reload`
4. Run the Streamlit frontend:
   - `streamlit run frontend/streamlit_app.py`
5. (Optional) Run with Docker Compose:
   - `docker-compose up --build`
