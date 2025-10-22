# 🛡️ ETLArmor: End-to-End MLOps Network Security System

### 🚀 Overview
**ETLArmor** is an **end-to-end MLOps project** designed to detect and defend against **network security threats** using **automated ETL pipelines** and **machine learning models**.  
It demonstrates a full production workflow — from **data ingestion**, **validation**, **transformation**, and **model training**, to **deployment** and **monitoring**.

---

### 🧠 Key Features
- ⚙️ **Automated ETL Pipeline** – Handles data extraction, transformation, and loading from network traffic sources.  
- 🧪 **Data Validation & Drift Detection** – Ensures quality and consistency using YAML-based schema validation.  
- 🧩 **Modular MLOps Architecture** – Clean, component-based Python modules.  
- 📊 **Experiment Tracking** – MLflow integration for tracking runs and metrics.  
- ☁️ **Cloud Sync Support** – Optional S3 sync for model artifacts and logs.  
- 🧠 **Model Training & Evaluation** – Supports multiple ML algorithms for classification.  
- 🧾 **Dockerized Deployment** – Simplifies reproducibility and deployment.  

---

### 🧩 Tech Stack
- **Language:** Python 3.12  
- **Frameworks/Libraries:** scikit-learn, pandas, numpy, mlflow, flask  
- **MLOps Tools:** MLflow, Docker  
- **Cloud (Optional):** AWS S3  
- **Logging & Monitoring:** Python logging, MLflow UI  

---

### ⚙️ How to Run

#### 1️⃣ Clone the repository
```bash
git clone https://github.com/Vankydwivedi/ETLArmor.git
cd ETLArmor
2️⃣ Create a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # On Windows
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the training pipeline
bash
Copy code
python main.py
5️⃣ Launch web app (Flask)
bash
Copy code
python app.py
Then open http://127.0.0.1:5000/ in your browser.

📊 MLflow Tracking
To view experiments locally:

bash
Copy code
mlflow ui
Then visit http://127.0.0.1:5000 or http://localhost:5000 for the dashboard.

📦 Deployment (Docker)
Build and run using Docker:

bash
Copy code
docker build -t etlarmor .
docker run -p 5000:5000 etlarmor
📈 Results
The model achieves strong performance on phishing/network intrusion detection tasks with:

High accuracy

Low false-positive rate

Stable metrics across validation datasets

(Add specific accuracy or confusion matrix once finalized)

🧰 Future Enhancements
Integration with real-time network stream ingestion

CI/CD setup using GitHub Actions

Model registry with MLflow

API Gateway + Streamlit Dashboard

