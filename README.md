# Sales & Demand Forecasting Dashboard 📈

This repository contains a **Sales and Demand Forecasting System** built with Python and Streamlit. The system predicts future sales using historical business data and presents the results in a clear, business-friendly dashboard designed for store owners, startup founders, and business managers.

This project was built as part of the **Future Interns** Machine Learning task.

## 🚀 Features

- **Interactive Web Dashboard**: Built with Streamlit for a seamless user experience.
- **Data Preprocessing**: Handles missing values and automatically fills gaps in time-series data.
- **Feature Engineering**: Extracts time-based features like `year`, `month`, `day`, `day of the week`, and `is_weekend` to capture seasonality and trends.
- **Machine Learning Forecasting**: Utilizes a `RandomForestRegressor` from `scikit-learn` to learn from historical patterns and predict future demand.
- **Model Evaluation**: Displays actionable error metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Error Percentage (MAPE).
- **Business Insights**: Translates raw data into plain English, offering concrete suggestions for inventory management, staffing, and cash flow based on the model's predictions.
- **Future Forecasting**: Plots a clear 30-day projection of future sales.

## 🛠️ Tech Stack

- **Python**
- **Streamlit**: For the interactive UI.
- **Pandas & NumPy**: For data manipulation and feature engineering.
- **Scikit-Learn**: For the machine learning forecasting model.
- **Matplotlib & Seaborn**: For rich, business-friendly visualizations.

## 📥 Dataset

The project uses historical store sales data (e.g., Kaggle Store Sales dataset). Note that the raw dataset (`train.csv`) is excluded from the repository to comply with GitHub's file size limits, but can be added locally.

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Akarshakk/FUTURE_ML_01.git
   cd FUTURE_ML_01
   ```

2. Make sure you have the required `train.csv` dataset in the root directory.

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. The dashboard will automatically open in your default browser at `http://localhost:8501`.

## 📌 Deliverables Overview

✔ **Data cleaning and handling missing values**
✔ **Time-based feature engineering**
✔ **Forecasting using regression**
✔ **Model evaluation and error analysis**
✔ **Business-friendly forecast visualizations**

---
*Built for Future Interns. Check them out on [LinkedIn](https://www.linkedin.com/company/future-interns/).*
