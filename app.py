import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import altair as alt

# Path to the folder containing the models
models_path = 'https://raw.githubusercontent.com/doringber1996/final_project/main/'

# Load the dataset containing model information
predictions_df = pd.read_csv('https://raw.githubusercontent.com/doringber1996/final_project/main/predictions_df.csv')  

# Define the list of dishes
dish_columns = predictions_df['Dish'].unique()

# Preprocessing function
def preprocess_input(start_date, end_date, num_customers):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'Date': dates})
    data['יום בשבוע'] = data['Date'].dt.dayofweek + 1
    data['חודש'] = data['Date'].dt.month
    data['מספר לקוחות מנורמל'] = num_customers
    # יצירת אובייקט MinMaxScaler
    scaler = MinMaxScaler()
    # החלת MinMaxScaler על עמודת 'מספר לקוחות מנורמל'
    data['מספר לקוחות מנורמל'] = scaler.fit_transform(data[['מספר לקוחות מנורמל']])

    return data

# Prediction function
def predict_dishes(start_date, end_date, num_customers):
    results = {}
    input_data = preprocess_input(start_date, end_date, num_customers)

    for dish in dish_columns:
        best_model_type = predictions_df.loc[predictions_df['Dish'] == dish, 'Model'].values[0]
        predictions = load_model_and_predict(dish, input_data, best_model_type)
        results[dish] = predictions

    return results

# Define function to load model and make predictions
def load_model_and_predict(dish, input_data, model_type):
    model_type = model_type.lower()
    if model_type == 'elastic net':
        model_type = 'en'
    elif model_type == 'xgboost':
        model_type = 'xgb'
    elif model_type == 'random forest':
        model_type = 'rf'
    elif model_type == 'holt-winters':
        model_type = 'hw'
    elif model_type == 'arima':
        model_type = 'arima'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_file = os.path.join(models_path, f'best_{model_type}_model_{dish}.pkl')
    if not os.path.isfile(model_file):
        st.error(f"Model file not found: {model_file}")
        return np.array([])

    model = joblib.load(model_file)

    if model_type == 'arima':
        predictions = model.forecast(steps=len(input_data))
    elif model_type in ['holt-winters', 'hw']:
        predictions = model.forecast(steps=len(input_data))
    elif model_type in ['elastic net', 'en']:
        features = input_data[['יום בשבוע', 'חודש', 'מספר לקוחות מנורמל']]
        predictions = model.predict(features)
    elif model_type in ['xgboost', 'xgb']:
        features = input_data[['יום בשבוע', 'חודש', 'מספר לקוחות מנורמל']]
        predictions = model.predict(features)
    elif model_type in ['random forest', 'rf']:
        features = input_data[['יום בשבוע', 'חודש', 'מספר לקוחות מנורמל']]
        predictions = model.predict(features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # המרה למספרים שלמים בעזרת np.ceil
    predictions = np.ceil(predictions).astype(int)

    return predictions

# Streamlit GUI
st.title("Dish Prediction")

st.header("Input Parameters")

# Input fields
start_date = st.date_input("Start Date", datetime.now())
end_date = st.date_input("End Date", datetime.now() + timedelta(days=1))
num_customers = st.number_input("Number of Customers", min_value=1, step=1)

if st.button("Predict"):
    results = predict_dishes(start_date, end_date, num_customers)

    results_text = "Predicted Dishes:\n"
    predictions_data = []
    for dish, prediction in results.items():
        results_text += f"{dish}: {prediction.sum()}\n"
        predictions_data.append({"Dish": dish, "Prediction": prediction.sum()})

    st.text(results_text)

    # Display results as a table
    st.subheader("Prediction Results")
    predictions_df = pd.DataFrame(predictions_data)
    st.table(predictions_df)

    # Display results as a bar chart
    st.subheader("Prediction Bar Chart")
    chart = alt.Chart(predictions_df).mark_bar().encode(
        x='Dish',
        y='Prediction',
        tooltip=['Dish', 'Prediction']
    ).properties(width=700, height=400)
    st.altair_chart(chart)

    # Provide option to download results
    st.subheader("Download Results")
    csv = predictions_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='predictions.csv', mime='text/csv')
