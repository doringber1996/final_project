import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import altair as alt
from urllib.parse import quote
from io import BytesIO
import requests

# Path to the folder containing the models and images
models_path = 'https://raw.githubusercontent.com/doringber1996/final_project/main/'

# Load the dataset containing model information
predictions_df = pd.read_csv(f'{models_path}predictions_df.csv')

# Load images from GitHub
logo_url = f'{models_path}logo.png'
restaurant_url = f'{models_path}cafe-italia.jpg'

# Define the list of dishes
dish_columns = predictions_df['Dish'].unique()

# Preprocessing function
def preprocess_input(start_date, end_date, num_customers):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'Date': dates})
    data.set_index('Date', inplace=True)  # הגדרת אינדקס התאריכים
    data['יום בשבוע'] = data.index.dayofweek + 1
    data['חודש'] = data.index.month
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

    # החלפת התו ' בתו _ בשם הקובץ
    sanitized_dish = dish.replace("'", "_")
    encoded_dish = quote(sanitized_dish)
    model_file = f'{models_path}best_{model_type}_model_{encoded_dish}.pkl'
    
    # Download the model file from the given URL
    try:
        response = requests.get(model_file)
        response.raise_for_status()  # Check if the request was successful
        model = joblib.load(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Model file not found or error in loading: {model_file}, Error: {e}")
        return np.array([])
    except Exception as e:
        st.error(f"Error in loading the model: {model_file}, Error: {e}")
        return np.array([])

    if model_type == 'arima':
        predictions = model.forecast(steps=len(input_data))
    elif model_type in ['holt-winters', 'hw']:
        predictions = model.forecast(steps=len(input_data))
    else:
        features = input_data[['יום בשבוע', 'חודש', 'מספר לקוחות מנורמל']]
        predictions = model.predict(features)

    # המרה למספרים שלמים בעזרת np.ceil
    predictions = np.ceil(predictions).astype(int)

    return predictions

st.markdown(
    f"""
    <style>
    .main {{
        background-image: url("{restaurant_url}");
        background-size: cover;
        position: relative;
        z-index: 1;
        color: white;
    }}
    .main:before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5); /* שכבה כהה שקופה */
        z-index: -1;
    }}
    .stButton button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
    }}
    .sttitle, .stheader, .stSubheader, .stMarkdown, .stText, .stNumberInput label, .stDateInput label {{
        color: white !important;
    }}
    .stTextInput, .stNumberInput input {{
        color: black;
    }}
    .css-10trblm, .css-1v3fvcr p {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.image(logo_url, width=200, use_column_width=False)

st.title("Dish Prediction Application")

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
        x=alt.X('Dish', sort=None, axis=alt.Axis(labelAngle=-30)),  # שינוי הזווית של התוויות בגרף ל-30 מעלות
        y='Prediction',
        color=alt.Color('Dish', scale=alt.Scale(scheme='tableau20')),  # צבעים ייחודיים לכל מנה
        tooltip=['Dish', 'Prediction']
    ).properties(width=700, height=400)
    st.altair_chart(chart)

    # Provide option to download results
    st.subheader("Download Results")
    csv = predictions_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='predictions.csv', mime='text/csv')
