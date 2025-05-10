# Customer Account Status Prediction: ML Model Comparison & Deployment

This project leverages a comprehensive machine learning pipeline to predict the current status of bank customer accounts. Using a dataset enriched with demographic, behavioral, and transactional features, multiple classification algorithms are compared to identify the most accurate and robust model. The project includes full preprocessing, outlier handling, feature encoding, model training, evaluation, and deployment with Streamlit.

## 📊 Features
- Cleans and preprocesses real-world financial customer data
- Compares 12+ models including Random Forest, XGBoost, Neural Networks, and Ensemble methods
- Visual comparison of model performance
- Saves the best-performing model using `joblib`
- Deploys with a user-friendly Streamlit web interface for live predictions

## 🚀 How to Run

### 1. Clone this repo
```bash
git clone https://github.com/yourusername/ML-Customer-Status-Prediction.git
cd ML-Customer-Status-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
streamlit run app.py
```

## 📁 Files
- `Data_TarImp_CleanOutlier.csv`: Cleaned dataset
- `model_training.py`: Script for data processing, model training & comparison
- `best_model.pkl`: Saved trained model
- `app.py`: Streamlit app
- `requirements.txt`: All dependencies

---

📄 requirements.txt

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit

---
# Road map 
![Image](https://github.com/user-attachments/assets/a05b6abd-b2f9-4d64-bf6f-7938af1cac05)

