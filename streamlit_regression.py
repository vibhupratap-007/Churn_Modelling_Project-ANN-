import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

st.set_page_config(page_title="Estimated Salary Prediction", layout="centered")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('regression_model.h5')

    with open('label_encoder_gender.pkl','rb') as f:
        label_encoder_gender = pickle.load(f)

    with open('onehot_encoder_geo.pkl','rb') as f:
        onehot_encoder_geo = pickle.load(f)

    with open('scaler.pkl','rb') as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

st.title('Estimated Salary Prediction')


col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0].tolist())
    gender = st.selectbox('Gender', label_encoder_gender.classes_.tolist())
    age = st.slider('Age', 18, 92, 30)
    balance = st.number_input('Balance', value=0.0, format="%.2f")
    credit_score = st.number_input('Credit Score', value=600.0, format="%.2f")

with col2:
    tenure = st.slider('Tenure', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

if st.button("Predict Estimated Salary"):
    try:
        
        input_df = pd.DataFrame({
            'CreditScore': [credit_score],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
        })

        # Encode gender as during training
        gender_encoded_val = label_encoder_gender.transform([gender])[0]
        input_df['Gender'] = gender_encoded_val

        # One-hot encode geography and append resulting columns
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
        geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)
        input_df = pd.concat([input_df.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

        # Align columns to what the scaler expects (fill missing numeric columns with 0)
        expected = getattr(scaler, "feature_names_in_", None)
        if expected is None:
            if 'EstimatedSalary' not in input_df.columns:
                input_df['EstimatedSalary'] = 0.0
            input_aligned = input_df  # best-effort fallback
        else:
            # ensure EstimatedSalary exists if scaler expects it
            if 'EstimatedSalary' in expected and 'EstimatedSalary' not in input_df.columns:
                input_df['EstimatedSalary'] = 0.0
            input_aligned = input_df.reindex(columns=expected, fill_value=0)

        # Transform and predict
        input_scaled = scaler.transform(input_aligned)
        pred = model.predict(input_scaled)

        # model output -> single float
        predicted_salary = float(pred[0][0]) if np.ndim(pred) == 2 else float(pred[0])
        st.success(f'Predicted Estimated Salary: ${predicted_salary:,.2f}')

    except Exception as e:
        st.error("Prediction failed. See error message below.")
        st.exception(e)
