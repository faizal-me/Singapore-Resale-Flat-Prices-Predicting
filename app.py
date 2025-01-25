import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Singapore Resale Flat Price Prediction",
    page_icon="🏠", 
    layout="wide")

# Access the CSV file from the local system
csv_file_path = 'finaldf.csv'
df = pd.read_csv(csv_file_path)

# Define numerical and categorical columns
numerical_cols = ['floor_area_sqm', 'remaining_lease', 'lower_storey_range', 'upper_storey_range']
categorical_cols = ['town', 'flat_type', 'block', 'street_name', 'flat_model']
additional_cols = ['lease_commence_date', 'transaction_year', 'transaction_month']

# Initialize label encoders and reverse mappings
label_encoders = {}
reverse_mappings = {}
for column in categorical_cols:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    reverse_mappings[column] = dict(enumerate(le.classes_))  # For original name display

# Define features (X) and target (y)
X = df.drop(columns=['resale_price'])
y = df['resale_price']

# Define the preprocessor and pipeline
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numerical_cols)],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Fit the pipeline to the entire dataset
pipeline.fit(X, y)

# Streamlit Application
st.title('🏠 Singapore Resale Flat Price Prediction')

# Side menu for section selection with emojis
st.sidebar.title("MENU")
section = st.sidebar.radio("Select a section", [
    "🏠 Home", 
    "📍 Location Details", 
    "🏢 Flat Specifications", 
    "🗓️ Transaction Details", 
    "💵 Predict Resale Price"
])

# Initialize session state for storing input values
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'town': '',
        'flat_type': '',
        'block': '',
        'street_name': '',
        'flat_model': '',
        'floor_area_sqm': 100,
        'remaining_lease': 80,
        'lower_storey_range': 5,
        'upper_storey_range': 10,
        'lease_commence_date': datetime(2000, 1, 1),
        'transaction_year': 2024,
        'transaction_month': 1
    }

if section == "🏠 Home":
    st.image("Singapore.webp", caption="Singapore", use_column_width=True)  
    st.markdown("### About This Application")
    st.write(
        """
        This application helps you predict the resale price of flats in Singapore based on various features such as location, flat specifications, and transaction details.
        
        Use the sidebar to navigate through different sections to input the details of the flat and get an estimated resale price.
        """
    )

elif section == "📍 Location Details":
    st.markdown('<div style="font-size: 24px; font-weight: bold;">Location Details</div>', unsafe_allow_html=True)
    st.session_state.input_values['town'] = st.selectbox("Select the town where the flat is located", reverse_mappings['town'].values())
    st.session_state.input_values['flat_type'] = st.selectbox("Select the flat type", reverse_mappings['flat_type'].values())
    st.session_state.input_values['block'] = st.selectbox("Choose the block of the flat", reverse_mappings['block'].values())
    st.session_state.input_values['street_name'] = st.selectbox("Select the street name", reverse_mappings['street_name'].values())
    st.session_state.input_values['flat_model'] = st.selectbox("Choose the flat model", reverse_mappings['flat_model'].values())

elif section == "🏢 Flat Specifications":
    st.markdown('<div style="font-size: 24px; font-weight: bold;">Flat Specifications</div>', unsafe_allow_html=True)
    st.session_state.input_values['floor_area_sqm'] = st.number_input("Enter the floor area of the flat (sqm)", min_value=20, max_value=200, value=st.session_state.input_values['floor_area_sqm'])
    st.session_state.input_values['remaining_lease'] = st.number_input("Number of years remaining on the lease (years)", min_value=1, max_value=99, value=st.session_state.input_values['remaining_lease'])
    st.session_state.input_values['lower_storey_range'] = st.number_input("Enter the lower bound of the storey range", min_value=1, max_value=50, value=st.session_state.input_values['lower_storey_range'])
    st.session_state.input_values['upper_storey_range'] = st.number_input("Enter the upper bound of the storey range", min_value=1, max_value=50, value=st.session_state.input_values['upper_storey_range'])

elif section == "🗓️ Transaction Details":
    st.markdown('<div style="font-size: 24px; font-weight: bold;">Transaction Details</div>', unsafe_allow_html=True)
    st.session_state.input_values['lease_commence_date'] = st.date_input("Select the lease commence date", value=st.session_state.input_values['lease_commence_date'])
    st.session_state.input_values['transaction_year'] = st.number_input("Enter the year of the resale", min_value=2000, max_value=2024, value=st.session_state.input_values['transaction_year'])
    st.session_state.input_values['transaction_month'] = st.number_input("Enter the month of the resale", min_value=1, max_value=12, value=st.session_state.input_values['transaction_month'])

elif section == "💵 Predict Resale Price":
    st.markdown('<div style="font-size: 24px; font-weight: bold;">Predict Resale Price</div>', unsafe_allow_html=True)

    # Encode the selected categorical values
    def encode_input(value, encoder):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            st.error(f"Unrecognized value: {value}. Please select a valid option.")
            st.stop()

    town_encoded = encode_input(st.session_state.input_values['town'], label_encoders['town'])
    flat_type_encoded = encode_input(st.session_state.input_values['flat_type'], label_encoders['flat_type'])
    block_encoded = encode_input(st.session_state.input_values['block'], label_encoders['block'])
    street_name_encoded = encode_input(st.session_state.input_values['street_name'], label_encoders['street_name'])
    flat_model_encoded = encode_input(st.session_state.input_values['flat_model'], label_encoders['flat_model'])

    # Create input dataframe
    input_data = pd.DataFrame({
        'town': [town_encoded],
        'flat_type': [flat_type_encoded],
        'block': [block_encoded],
        'street_name': [street_name_encoded],
        'flat_model': [flat_model_encoded],
        'floor_area_sqm': [st.session_state.input_values['floor_area_sqm']],
        'remaining_lease': [st.session_state.input_values['remaining_lease']],
        'lower_storey_range': [st.session_state.input_values['lower_storey_range']],
        'upper_storey_range': [st.session_state.input_values['upper_storey_range']],
        'lease_commence_date': [datetime.now().year - st.session_state.input_values['lease_commence_date'].year],
        'transaction_year': [st.session_state.input_values['transaction_year']],
        'transaction_month': [st.session_state.input_values['transaction_month']]
    })

    # Button with loading spinner
    if st.button('🔍 Predict Resale Price'):
        with st.spinner('Making prediction...'):
            predicted_price = pipeline.predict(input_data)[0]
            st.success(f"🎉 Predicted Resale Price: ${predicted_price:.2f}")

        # Download button for result
        download_df = input_data.copy()
        download_df['predicted_resale_price'] = predicted_price
        csv = download_df.to_csv(index=False)
        st.download_button(label="Download Prediction Results", data=csv, file_name='resale_price_prediction.csv', mime='text/csv')
