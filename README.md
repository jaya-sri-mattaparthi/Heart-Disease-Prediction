# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease in patients based on various medical attributes. The project includes data analysis, model training, and a web-based interface for real-time predictions.

## 🔗 Live Demo
**[Try the Heart Disease Predictor](https://heart-disease-prediction-9k9o.onrender.com/)**

## 📊 Dataset Information

The dataset is sourced from the [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and dates from 1988. It consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V.

**Dataset Features:**
- **Total Attributes:** 76 (using 14 key features for prediction)
- **Target Variable:** Heart Disease (0 = No Disease, 1 = Disease)
- **Patient Privacy:** Names and SSNs replaced with dummy values

### Attribute Details

| Feature | Description | Values/Range |
|---------|-------------|--------------|
| `age` | Age in years | Numerical |
| `sex` | Gender | 1 = Male, 0 = Female |
| `cp` | Chest Pain Type | 0 = Typical Angina<br>1 = Atypical Angina<br>2 = Non-anginal Pain<br>3 = Asymptomatic |
| `trestbps` | Resting Blood Pressure | mm Hg |
| `chol` | Serum Cholesterol | mg/dl |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Resting ECG Results | 0 = Normal<br>1 = ST-T Wave Abnormality<br>2 = Left Ventricular Hypertrophy |
| `thalach` | Maximum Heart Rate Achieved | Numerical |
| `exang` | Exercise Induced Angina | 1 = Yes, 0 = No |
| `oldpeak` | ST Depression (Exercise vs Rest) | Numerical |
| `slope` | Peak Exercise ST Segment Slope | Numerical |
| `ca` | Major Vessels Colored by Fluoroscopy | 0-3 |
| `thal` | Thalassemia | 0 = Normal<br>1 = Fixed Defect<br>2 = Reversible Defect |
| `target` | **Heart Disease Presence** | **1 = Yes, 0 = No** |

## 🚀 Project Workflow

### 1. Data Collection & Preprocessing
- Imported necessary Python libraries
- Loaded and cleaned the heart disease dataset
- Handled missing values and data inconsistencies

### 2. Exploratory Data Analysis (EDA)
- Analyzed feature distributions
- Explored correlations between variables
- Visualized patterns in the data

### 3. Model Development
- **Data Splitting:** Train-test split for model validation
- **Feature Scaling:** Normalized features for optimal performance
- **Model Training:** 
  - **Logistic Regression:** Achieved 81% accuracy
  - **Random Forest:** Achieved 98% accuracy ⭐

### 4. Model Deployment
- Saved the best-performing Random Forest model
- Created a Flask web application with user-friendly interface
- Deployed on Render for public access

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Flask** - Web application framework
- **HTML/CSS** - Frontend interface
- **Render** - Cloud deployment platform

## 📈 Model Performance

| Algorithm | Accuracy | Status |
|-----------|----------|---------|
| Logistic Regression | 81% | ✅ Good |
| **Random Forest** | **98%** | ⭐ **Selected** |

*Random Forest was chosen for deployment due to its superior performance.*

## 🖥️ Usage

### Online Usage
Simply visit the [live application](https://heart-disease-prediction-9k9o.onrender.com/) and input the required medical parameters to get an instant heart disease risk prediction.

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
heart-disease-prediction/
│
├── Dataset/
    └── heart.csv 
├── Notebook/
│   └── heart_disease_prediction.ipynb   # Jupyter notebook with EDA
├── models/
    └── rf.pkl       # Trained Random Forest model
    └── scaler.pkl  
├── templates/
│   └── index.html       # Web interface
├── app.py                 # Flask application
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🔬 Key Insights

- Random Forest significantly outperformed Logistic Regression (98% vs 81%)
- Feature engineering and proper scaling improved model accuracy
- The model successfully identifies high-risk patients for heart disease

## ⚠️ Medical Disclaimer

This application is for **educational and research purposes only**. It should **not be used as a substitute for professional medical advice, diagnosis, or treatment**. Always consult with qualified healthcare providers for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


**Made with ❤️ for healthcare innovation**

*If you found this project helpful, please consider giving it a ⭐ on GitHub!*
