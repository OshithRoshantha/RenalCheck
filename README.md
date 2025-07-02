# RenalCheck 🩺

A comprehensive machine learning-based kidney disease risk detection system that provides accurate risk assessment using clinical and laboratory parameters.

## 🎯 Project Overview

RenalCheck is an advanced data science project that leverages machine learning algorithms to predict kidney disease risk levels in patients. The system analyzes 25+ clinical features including blood parameters, urine tests, and patient demographics to classify patients into five risk categories: No Disease, Low Risk, Moderate Risk, High Risk, and Severe Disease.

## 📁 Project Structure

```
RenalCheck/
├── app/
│   └── main.py                 # Streamlit web application
├── data/
│   ├── raw/
│   │   └── kidney_disease_dataset.csv
│   └── processed/
│       ├── cleanedData.csv
│       └── processedData.csv
├── model/
│   ├── bestModel.pkl          # Trained model
│   ├── labelEncoder.pkl       # Label encoder
│   ├── scaler.pkl            # Feature scaler
│   └── metadata.pkl          # Model metadata
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── Feature_Engineering.ipynb
│   ├── Preprocessing.ipynb
│   └── Train_Evaluate.ipynb  # Model training & evaluation
├── reports/
│   └── figures/              # Generated visualizations
└── README.md
```

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OshithRoshantha/RenalCheck
   cd RenalCheck
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost torch matplotlib seaborn joblib imbalanced-learn tqdm
   ```

## 🚀 Usage

### Running the Web Application

1. **Navigate to the app directory**:
   ```bash
   cd app
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

3. **Access the application**:
   Open your browser and go to `http://localhost:8501`


