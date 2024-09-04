#Singapore Resale Flat Price Prediction Application 🏠
This Streamlit application predicts the resale prices of flats in Singapore based on various features such as location, flat specifications, and transaction details. The predictions are powered by a machine learning model built using sklearn's DecisionTreeRegressor.

#Features 📋
📍 Location Details: Input fields for town, flat type, block, street name, and flat model.
🏢 Flat Specifications: Inputs for floor area, remaining lease, and storey range.
🗓️ Transaction Details: Inputs for lease commencement date and resale transaction date (year and month).
💵 Predict Resale Price: Based on the inputs, the model will predict the resale price of the flat. You can also download the prediction results.
#How It Works ⚙️
##Data Preprocessing:

Categorical columns (town, flat_type, block, street_name, flat_model) are label encoded.
Numerical columns (floor_area_sqm, remaining_lease, lower_storey_range, upper_storey_range) are standardized using StandardScaler.
The lease_commence_date is transformed to calculate the age of the flat.
The selected transaction year and month represent the date of the resale.
##Model:

The machine learning model used is DecisionTreeRegressor from sklearn.
The model is trained on historical Singapore flat resale data, accessible via the local CSV file finaldf.csv.
##User Interface:

Users can navigate through different sections (Home, Location Details, Flat Specifications, Transaction Details) via the sidebar.
Inputs from the user are processed and transformed into the correct format for prediction by the machine learning model.
The application displays the predicted resale price, and the user can download the results as a CSV file.
