import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model_ann.h5')

# Load the transformer(encoders and scaler)
with open('feature_transformer.pkl', 'rb') as file:
    feature_transformer = pickle.load(file)


## streamlit app
#st.title('Customer Churn Prediction')

# geography_options = ['France', 'Spain', 'Germany']
# gender_options = ['Male', 'Female']

# User input
# Page configuration
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 6rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FFFFFF 0%, #FFFF00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        padding-left: 10px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏦🏦 Customer Churn Predictor 🏦🏦</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict customer retention with AI-powered insights</p>', unsafe_allow_html=True)

st.divider()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">📋 Personal Information</p>', unsafe_allow_html=True)
    
    geography = st.selectbox(
        '🌍 Geography',
        ['France', 'Spain', 'Germany'],
        help="Select the customer's country"
    )
    
    gender = st.selectbox(
        '👤 Gender',
        ['Male', 'Female'],
        help="Select the customer's gender"
    )
    
    age = st.slider(
        '🎂 Age',
        18, 92, 35,
        help="Customer's age in years"
    )
    
    tenure = st.slider(
        '⏱️ Tenure (Years)',
        0, 10, 5,
        help="Number of years as a customer"
    )

with col2:
    st.markdown('<p class="section-header">💰 Financial Information</p>', unsafe_allow_html=True)
    
    credit_score = st.number_input(
        '💳 Credit Score',
        min_value=300,
        max_value=850,
        value=650,
        step=10,
        help="Credit score ranging from 300 to 850"
    )
    
    balance = st.number_input(
        '💵 Account Balance',
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.2f",
        help="Current account balance"
    )
    
    estimated_salary = st.number_input(
        '💼 Estimated Salary',
        min_value=0.0,
        value=75000.0,
        step=5000.0,
        format="%.2f",
        help="Annual estimated salary"
    )

# Full width section for products and services
st.markdown('<p class="section-header">🛍️ Products & Services</p>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    num_of_products = st.slider(
        '📦 Number of Products',
        1, 4, 2,
        help="Total number of bank products held"
    )

with col4:
    has_cr_card = st.selectbox(
        '💳 Has Credit Card',
        options=[1, 0],
        format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
        help="Does the customer have a credit card?"
    )

with col5:
    is_active_member = st.selectbox(
        '🔄 Active Member',
        options=[1, 0],
        format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
        help="Is the customer actively using services?"
    )

st.divider()

# Prediction button
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
with col_button2:
    predict_button = st.button('🚀 Predict Churn Risk', use_container_width=True)

# Display input summary in an expandable section
with st.expander("📊 View Input Summary"):
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Age", f"{age} years")
        st.metric("Tenure", f"{tenure} years")
        st.metric("Products", num_of_products)
    
    with summary_col2:
        st.metric("Credit Score", credit_score)
        st.metric("Balance", f"${balance:,.2f}")
        st.metric("Geography", geography)
    
    with summary_col3:
        st.metric("Salary", f"${estimated_salary:,.2f}")
        st.metric("Credit Card", "Yes" if has_cr_card == 1 else "No")
        st.metric("Active Member", "Yes" if is_active_member == 1 else "No")

# Prediction results with YOUR model
if predict_button:
    with st.spinner('🔮 Analyzing customer data...'):
        # Step 1: Prepare input data in DataFrame format
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        
        # Step 2: Transform input data using your feature transformer
        input_array = feature_transformer.transform(input_data)
        
        # Step 3: Make prediction using your trained model
        prediction = model.predict(input_array)
        prediction_proba = prediction[0][0]
        
        # Convert to percentage
        churn_probability = prediction_proba * 100
        
        # Determine risk level
        if prediction_proba > 0.7:
            risk_level = "High"
            risk_color = "🔴"
        elif prediction_proba > 0.4:
            risk_level = "Medium"
            risk_color = "🟡"
        else:
            risk_level = "Low"
            risk_color = "🟢"
    
    # Display results
    st.divider()
    
    # Main result message
    if prediction_proba > 0.5:
        st.error(f"⚠️ **Alert:** The customer is likely to churn!")
    else:
        st.success(f"✅ **Good News:** The customer is likely to stay!")
    
    # Detailed metrics
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability:.1f}%",
            delta=f"{churn_probability - 50:.1f}% from baseline",
            delta_color="inverse"
        )
    
    with result_col2:
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level}",
        )
    
    with result_col3:
        confidence = (1 - abs(prediction_proba - 0.5) * 2) * 100
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.1f}%"
        )
    
    # Additional insights
    with st.expander("💡 View Detailed Insights"):
        st.write("**Prediction Details:**")
        st.write(f"- Raw probability score: {prediction_proba:.4f}")
        st.write(f"- Classification threshold: 0.5")
        st.write(f"- Decision: {'CHURN' if prediction_proba > 0.5 else 'STAY'}")
        
        st.write("\n**Risk Interpretation:**")
        if prediction_proba > 0.7:
            st.write("🔴 **High Risk:** Immediate retention action recommended")
        elif prediction_proba > 0.4:
            st.write("🟡 **Medium Risk:** Monitor customer engagement closely")
        else:
            st.write("🟢 **Low Risk:** Customer appears satisfied and stable")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>💡 Powered by Machine Learning | Built with Streamlit</p>
        <p>Developed by Piyush Yadav</p>
    </div>
""", unsafe_allow_html=True)