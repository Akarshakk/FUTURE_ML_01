import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page configuration
st.set_page_config(page_title="Sales & Demand Forecasting", page_icon="📈", layout="wide")

st.title("📈 Business Sales & Demand Forecasting")
st.markdown("""
Welcome to the Demand Forecasting Dashboard! This tool helps business owners and managers plan inventory and optimize sales strategies by predicting future demand based on historical trends.
""")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    # Load dataset with selected columns to optimize memory
    df = pd.read_csv("train.csv", usecols=["date", "store_nbr", "family", "sales", "onpromotion"])
    df['date'] = pd.to_datetime(df['date'])
    return df

with st.spinner("Loading historical sales data..."):
    data = load_data()

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("Filter Data")

# Store selection
store_list = ["All Stores"] + sorted(data['store_nbr'].unique().tolist())
selected_store = st.sidebar.selectbox("Select Store", store_list)

# Product Family selection
family_list = sorted(data['family'].unique().tolist())
selected_family = st.sidebar.selectbox("Select Product Category", family_list)

# --- 3. DATA FILTERING & AGGREGATION ---
if selected_store == "All Stores":
    filtered_data = data[data['family'] == selected_family]
    # Aggregate across all stores
    daily_data = filtered_data.groupby('date').agg({'sales': 'sum', 'onpromotion': 'sum'}).reset_index()
else:
    filtered_data = data[(data['store_nbr'] == selected_store) & (data['family'] == selected_family)]
    daily_data = filtered_data.groupby('date').agg({'sales': 'sum', 'onpromotion': 'sum'}).reset_index()

# Handle missing dates if any
daily_data.set_index('date', inplace=True)
daily_data = daily_data.asfreq('D', fill_value=0) # Fill missing days with 0 sales
daily_data.reset_index(inplace=True)

# --- 4. EXPLORATORY DATA ANALYSIS (EDA) ---
st.subheader(f"📊 Historical Sales for {selected_family} ({'All Stores' if selected_store == 'All Stores' else f'Store {selected_store}'})")

# Line chart of sales over time
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=daily_data, x='date', y='sales', ax=ax, color='blue', linewidth=1.5)
ax.set_title("Daily Sales Over Time", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Total Sales")
st.pyplot(fig)

# Feature Engineering for EDA & Modeling
daily_data['year'] = daily_data['date'].dt.year
daily_data['month'] = daily_data['date'].dt.month
daily_data['day'] = daily_data['date'].dt.day
daily_data['dayofweek'] = daily_data['date'].dt.dayofweek
daily_data['is_weekend'] = (daily_data['dayofweek'] >= 5).astype(int)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sales by Day of the Week**")
    fig_dow, ax_dow = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=daily_data, x='dayofweek', y='sales', ax=ax_dow, palette='Set2')
    ax_dow.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax_dow.set_xlabel("Day of Week")
    ax_dow.set_ylabel("Sales")
    st.pyplot(fig_dow)

with col2:
    st.markdown("**Sales vs. Promotions**")
    fig_promo, ax_promo = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=daily_data, x='onpromotion', y='sales', alpha=0.5, ax=ax_promo, color='purple')
    ax_promo.set_xlabel("Number of Items on Promotion")
    ax_promo.set_ylabel("Sales")
    st.pyplot(fig_promo)

# --- 5. FORECASTING MODEL ---
st.subheader("🔮 Demand Forecasting")
st.markdown("""
We use a **Random Forest Regressor** to learn from historical patterns (e.g., day of week, time of year, promotions) and predict future sales. 
The data is split chronologically: we train on past data and test on the last 30 days to see how well it performs.
""")

# Prepare features and target
features = ['month', 'day', 'dayofweek', 'is_weekend', 'onpromotion']
X = daily_data[features]
y = daily_data['sales']

# Chronological Train-Test Split (Last 30 days for testing)
test_days = 30
if len(daily_data) > test_days * 2:
    X_train, X_test = X[:-test_days], X[-test_days:]
    y_train, y_test = y[:-test_days], y[-test_days:]
    dates_train, dates_test = daily_data['date'][:-test_days], daily_data['date'][-test_days:]

    # Train model
    with st.spinner("Training forecasting model..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    avg_sales = y_test.mean()
    mape = (mae / avg_sales) * 100 if avg_sales > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", help="Average error in sales prediction.")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", help="Standard deviation of the residuals.")
    col3.metric("Error % (relative to average sales)", f"{mape:.2f}%", help="Error relative to average sales volume.")

    # Plot Actual vs Predicted for Test Set
    fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
    ax_pred.plot(dates_train[-60:], y_train[-60:], label='Historical Sales (Last 60 Days Training)', color='gray')
    ax_pred.plot(dates_test, y_test, label='Actual Sales (Test Set)', color='blue')
    ax_pred.plot(dates_test, predictions, label='Predicted Sales', color='red', linestyle='--')
    ax_pred.set_title("Actual vs. Predicted Sales (Last 30 Days)", fontsize=14)
    ax_pred.set_xlabel("Date")
    ax_pred.set_ylabel("Sales")
    ax_pred.legend()
    st.pyplot(fig_pred)

    # --- 6. BUSINESS INSIGHTS ---
    st.subheader("💡 Business Insights & Planning")
    st.markdown(f"""
    **What does this forecast mean?**
    - The model's predictions have an average error of **{mae:.2f} units** per day. This means that, on average, our forecast is off by this amount.
    - Given that the average sales in the last 30 days were **{avg_sales:.2f} units**, the forecast error is roughly **{mape:.2f}%**.
    
    **How can the business use this?**
    1. **Inventory Management:** Ensure you stock at least the predicted amount plus a safety buffer (e.g., the MAE of {mae:.2f} units) to avoid out-of-stock scenarios.
    2. **Staffing:** Schedule more staff on days with high predicted demand, which often correlates with weekends or promotional events as seen in the EDA charts.
    3. **Cash Flow:** Use predicted sales volumes to estimate revenue and manage weekly cash flow.
    """)

else:
    st.warning("Not enough data to train a forecasting model. Please select a different store or product family.")
