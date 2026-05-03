# 🚀 Delivery Time Prediction — End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow)
![ZenML](https://img.shields.io/badge/ZenML-Pipeline-7B42BC?style=flat-square)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square&logo=amazonaws)

---

## 📌 Overview

This project demonstrates a complete **end-to-end MLOps pipeline** for predicting delivery time using Machine Learning and Deep Learning.

It covers the full lifecycle:

```
Data → Model → Pipeline → API → UI → Deployment (AWS)
```

---

## 🎯 Problem Statement

Predict the **delivery time (in minutes)** based on:

| Feature | Description |
|---|---|
| `Distance` | Distance between restaurant and customer |
| `Weather` | Weather condition at the time of delivery |
| `Traffic Level` | Current traffic intensity |
| `Time of Day` | Morning / Afternoon / Evening / Night |
| `Vehicle Type` | Bike / Scooter / Car |
| `Preparation Time` | Time taken to prepare the order |
| `Courier Experience` | Experience level of the delivery agent |

---

## 🧠 Models Used

- ✅ **Linear Regression** — Baseline model
- ✅ **Neural Network (MLP)** — Built with PyTorch

> Best model is selected based on evaluation performance.

---

## 📊 Evaluation Metrics

Since this is a **regression problem**, the following metrics are used:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |

**Example Results:**
```
MAE  ≈ 5–7 minutes
RMSE ≈ 7–9 minutes
```

---

## 🏗️ Project Architecture

```
User → Streamlit UI → FastAPI → ML Model → Prediction
                          ↑
                        NGINX
                          ↑
                       AWS EC2
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **ML / DL** | PyTorch, Scikit-learn |
| **Pipeline** | ZenML |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **Deployment** | AWS EC2 |
| **Server** | NGINX |
| **Process Manager** | systemd |

---

## 📂 Project Structure

```
delivery-prediction-mlops/
│
├── app/
│   ├── api.py                  # FastAPI backend
│   └── streamlit_app.py        # Streamlit frontend
│
├── pipelines/
│   ├── training_pipeline.py    # ZenML pipeline definition
│   └── run_pipeline.py         # Pipeline runner
│
├── steps/
│   ├── ingest.py               # Data ingestion step
│   ├── preprocess.py           # Data preprocessing step
│   ├── train.py                # Model training step
│   └── evaluate.py             # Model evaluation step
│
├── model/
│   └── model.py                # Neural network architecture
│
├── data/
│   └── Food_Delivery_Times.csv # Dataset
│
├── model.pth                   # Saved PyTorch model weights
├── columns.pkl                 # Saved feature columns
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/delivery-prediction-mlops.git
cd delivery-prediction-mlops
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline
```bash
python -m pipelines.run_pipeline
```

### 5. Start the FastAPI Backend
```bash
uvicorn app.api:app --reload
```

### 6. Launch the Streamlit Frontend
```bash
streamlit run app/streamlit_app.py
```

> API docs available at: `http://127.0.0.1:8000/docs`

---

## 🌐 Deployment (AWS EC2)

- Deployed on **AWS Free Tier (EC2)**
- **NGINX** configured as a reverse proxy
- **systemd** used for auto-starting services on reboot

**Access the live app:**
```
http://<your-ec2-public-ip>/
```

---

## 💡 Key Learnings

- Building a model is only **20% of the work**
- Real value comes from:
  - 🚀 **Deployment** — making the model accessible
  - 📈 **Scalability** — handling real-world load
  - 🔒 **Reliability** — keeping services running
- Handling real-world challenges like:
  - Feature mismatch between training and inference
  - Model loading errors in production
  - Data preprocessing consistency across environments

---

## 🔥 Future Improvements

- [ ] 🐳 Docker containerization
- [ ] ⚙️ CI/CD pipeline with GitHub Actions
- [ ] 🔐 HTTPS + custom domain (SSL via Let's Encrypt)
- [ ] 📊 Model monitoring & drift detection
- [ ] 🗃️ Feature store integration

---

## 🙌 Author

**Nilesh Parmar**  
*Aspiring AI/ML Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/nileshpar835)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/nileshpar835)

---

## ⭐ Support

If you found this project helpful or interesting, please consider giving it a **⭐ star** on GitHub — it means a lot and helps others discover the project!

---

> *"The goal is to turn data into information, and information into insight."* — Carly Fiorina
