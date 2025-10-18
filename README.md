
# Conformal Health Risk Predictor

## Description

This project addresses the critical need for reliable uncertainty quantification in medical diagnoses. Traditional machine learning models often provide binary "Yes/No" predictions for conditions like heart disease without conveying the model's confidence, which can be risky in clinical settings.

This application leverages **Conformal Prediction** to build a more trustworthy health risk prediction system. It wraps a Random Forest classifier with a Conformal Prediction framework (using MAPIE) to not only predict the likelihood of heart disease but also to provide a mathematically guaranteed confidence interval for each prediction. This ensures that doctors and patients receive more transparent and reliable insights, helping to identify uncertain predictions that may require further attention.

The system is trained on the well-known **UCI Heart Disease Dataset**.

## Features

  * **Uncertainty Quantification**: Provides confidence intervals with predictions to measure certainty.
  * **Multiple Models**: Implements and compares Random Forest, Logistic Regression, SVM, and XGBoost.
  * **Interactive Web App**: A user-friendly interface built with Streamlit to input patient data and view predictions.
  * **Calibrated Confidence**: Delivers reliable prediction intervals with guaranteed coverage, ensuring that about 90% of predictions fall within their expected confidence bounds.

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ishaansahu22/Conformal-Health-Risk-Prediction.git
    cd Conformal-Health-Risk-Prediction
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the installation is complete, you can run the Streamlit web application:

```bash
streamlit run app.py
```

Navigate to the local URL provided by Streamlit in your web browser to start using the predictor.

## Project Structure

```
├── data/
│   └── heart.csv              # UCI Heart Disease Dataset
├── models/                    # Trained model files (.pkl)
├── notebooks/
│   ├── 01_explore.ipynb       # Exploratory Data Analysis
│   ├── 02_modeling.ipynb      # Model Training and Evaluation
│   └── 03_conformal.ipynb     # Conformal Prediction Implementation
├── src/
│   ├── preprocess.py          # Data preprocessing scripts
│   ├── train.py               # Model training scripts
│   └── conformal.py           # Conformal prediction logic
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
└── README.md                  # You are here!
```
