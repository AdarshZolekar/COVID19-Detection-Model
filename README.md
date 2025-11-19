# COVID-19 Detection Model

A Python-based machine learning project that predicts COVID-19 test results using patient symptoms and demographic data. The model analyzes various health indicators to classify coronavirus test outcomes with high accuracy using multiple ML algorithms.

---

## Features 

- Analyzes COVID-19 patient data from 2020-2021

- Data preprocessing with Label Encoding for categorical variables

- Multiple machine learning algorithms comparison

- Achieves 91.53% accuracy with Random Forest Classifier

- Comprehensive data visualization and analysis

- Training vs Testing performance comparison

- Built using Jupyter Notebook for interactive experimentation.

---

## How to Run 

1. **Clone the repository:**

```
   git clone https://github.com/AdarshZolekar/COVID19-Detection-Model.git
```

2. **Prepare the dataset:**

- Download or add the `covid_data_2020-2021.csv.zip` file to your Google Drive.

- Update the file path in the script to match your location.

3. **Run the script:** 

- Open a terminal or command prompt. 

- Navigate to the project directory:

```
   cd COVID19-Detection-Model
```

- Run the Jupyter Notebook:

```
   jupyter notebook Corona-Detection-Model.ipynb
```

- Or run the Python script directly:

```
   python Corona-Detection-Model.py
```

- Execute the notebook cells step-by-step to train models and visualize results.

---

## How It Works

1. Data Loading:
Reads the COVID-19 dataset containing patient symptoms and test results from 2020-2021.

2. Data Preprocessing:

- Removes unnecessary columns (test_date)

- Checks for duplicates and null values

- Converts categorical variables to numerical using Label Encoding

3. Feature Engineering:
Separates features (symptoms, age, gender) from target variable (corona_result).

4. Model Training & Comparison:
Tests multiple algorithms to find the best performer:
   - Logistic Regression (90.84% accuracy)
   - Random Forest Classifier (91.53% accuracy) Best
   - Naive Bayes Classifier
   - XGBoost Classifier (91.39% accuracy)

5. Visualization:
Generates plots showing predictions, error distribution and feature correlations.

---

## Dependencies

- pandas – Data handling and manipulation

- numpy – Numerical computations

- scikit-learn – Machine learning models and preprocessing

- xgboost – Gradient boosting classifier

- matplotlib – Data visualization

- seaborn – Statistical data visualization

- Jupyter Notebook – Interactive environment

Install them manually if needed:

```pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter```

---

## Dataset

- File: `covid_data_2020-2021.csv.zip`

- Description: Contains COVID-19 patient records with symptoms and test results

- Features:

  - `cough` – Presence of cough symptom (Yes/No)

  - `fever` – Presence of fever (Yes/No)

  - `sore_throat` – Sore throat symptom (Yes/No)

  - `shortness_of_breath` – Breathing difficulty (Yes/No)

  - `head_ache` – Headache symptom (Yes/No)

  - `age_60_and_above` – Age category (Yes/No)

  - `gender` – Patient gender (Male/Female)

  - `test_indication` – Reason for testing

  - `corona_result` – COVID-19 test result (Positive/Negative) **[Target Variable]**

---

## Results

**Model Performance Comparison:**

| Algorithm | Training Accuracy | Testing Accuracy |
|-----------|------------------|------------------|
| Logistic Regression | 90.84% | 90.84% |
| **Random Forest Classifier** | **~100%** | **91.53%** |
| Naive Bayes | Lower | Lower |
| XGBoost | ~100% | 91.39% |

**Best Model:** Random Forest Classifier with **91.53% test accuracy**

**Visualizations Include:**
- Actual vs Predicted comparison plots
- Error distribution charts
- Feature histograms
- Correlation scatter plots.

---

## Model Insights

The Random Forest Classifier provides the best balance between training and testing accuracy, indicating:

- Strong predictive capability for COVID-19 detection

- Good generalization to unseen data

- Effective use of symptom-based features

- Minimal overfitting compared to other models.

---

## Future Improvements

- Implement deep learning models (Neural Networks) for better accuracy

- Add cross-validation for more robust evaluation

- Include feature importance analysis

- Expand dataset with more recent COVID-19 variants

- Build a web interface using Flask or Streamlit for real-time predictions

- Add SHAP or LIME for model interpretability

- Implement ensemble methods combining multiple models.

---

## License

This project is open-source under the MIT License.

---

## Contributions

Contributions are welcome!

- Open an issue for bugs or feature requests

- Submit a pull request for improvements.

<p align="center">
  <a href="#top">
    <img src="https://img.shields.io/badge/%E2%AC%86-Back%20to%20Top-blue?style=for-the-badge" alt="Back to Top"/>
  </a>
</p>
