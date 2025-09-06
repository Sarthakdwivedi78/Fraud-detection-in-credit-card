import streamlit as st
import pickle
import pandas as pd

# --- 1. MODEL LOADING ---
@st.cache_data
def load_model(path):
    """Loads the pre-trained model from a .pkl file."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

model = load_model('fraud_model.pkl')

# --- 2. PRESET SCENARIO DATA ---
PRESETS = {
    "Typical Low-Value Purchase": {
        "Time": 96783.0, 'V1': -0.638, 'V2': -0.053, 'V3': 0.055, 'V4': -0.036, 'V5': -0.046, 
        'V6': 0.077, 'V7': -0.023, 'V8': -0.0007, 'V9': -0.057, 'V10': -0.053, 'V11': 0.006, 
        'V12': 0.003, 'V13': -0.013, 'V14': 0.020, 'V15': 0.081, 'V16': -0.044, 'V17': 0.028, 
        'V18': 0.006, 'V19': 0.009, 'V20': -0.020, 'V21': 0.060, 'V22': 0.017, 'V23': 0.024, 
        'V24': -0.002, 'V25': 0.036, 'V26': -0.061, 'V27': 0.005, 'V28': -0.027
    },
    "Known High-Risk/Fraudulent Pattern": {
        "Time": 80746.0, 'V1': -4.771, 'V2': 3.623, 'V3': -7.033, 'V4': 4.542, 'V5': -3.151, 
        'V6': -1.397, 'V7': -5.568, 'V8': 0.570, 'V9': -2.581, 'V10': -5.676, 'V11': 3.800, 
        'V12': -6.259, 'V13': -0.109, 'V14': -6.971, 'V15': -0.092, 'V16': -4.139, 'V17': -6.665, 
        'V18': -2.246, 'V19': 0.680, 'V20': 0.372, 'V21': 0.713, 'V22': 0.014, 'V23': -0.040, 
        'V24': -0.105, 'V25': 0.041, 'V26': 0.051, 'V27': 0.170, 'V28': 0.075
    }
}

# --- 3. APP LAYOUT ---
st.set_page_config(page_title="User-Friendly Fraud Detection", page_icon="🧑‍💻")
st.title("💡 User-Friendly Fraud Detection")

# --- NEW: OVERVIEW SECTION ---
with st.expander("What This Model Does (Overview)"):
    st.write("""
        This application uses a Machine Learning model to determine if a credit card transaction is likely fraudulent or legitimate.

        **How it Works:**
        - **Anonymous Data:** The model was trained on a dataset where sensitive transaction details were transformed into anonymous numerical features (`V1` to `V28`) using a technique called Principal Component Analysis (PCA) to protect user privacy.
        - **Pattern Recognition:** The model learns the complex patterns and relationships between these features that are characteristic of fraudulent activity.
        - **Prediction:** When you provide input, the model analyzes the pattern and calculates the probability of it being a fraudulent transaction based on what it learned from historical data.

        This app provides a simple interface to interact with the model by using preset scenarios that represent common transaction patterns.
    """)

# --- 4. USER-FRIENDLY INPUT ---
st.header("Transaction Details")
amount = st.number_input('Enter Transaction Amount ($)', min_value=0.0, value=100.0, step=10.0,
                         help="Enter the dollar amount of the transaction.")

st.subheader("Select a Transaction Scenario")
col1, col2 = st.columns(2)
with col1:
    if st.button("Typical Low-Value Purchase", use_container_width=True):
        st.session_state.preset = "Typical Low-Value Purchase"
with col2:
    if st.button("Suspicious/High-Risk Pattern", use_container_width=True):
        st.session_state.preset = "Known High-Risk/Fraudulent Pattern"

# --- 5. ADVANCED INPUT (HIDDEN BY DEFAULT) ---
with st.expander("Advanced Settings: Manual Feature Input"):
    st.write("These values are auto-filled by the scenario buttons above. You can adjust them manually if needed.")
    
    v_features = {}
    preset_name = st.session_state.get("preset", "Typical Low-Value Purchase")
    preset_values = PRESETS[preset_name]

    for i in range(1, 29):
        feature_name = f'V{i}'
        v_features[feature_name] = st.slider(feature_name, -50.0, 50.0, preset_values[feature_name], 0.1)

# --- 6. PREDICTION LOGIC ---
if model:
    if st.button('**Predict Transaction**', type="primary"):
        data = {'Time': preset_values['Time']}
        data.update(v_features)
        data['Amount'] = amount
        
        column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = pd.DataFrame(data, index=[0])[column_order]
        
        st.subheader("Model Input")
        st.dataframe(input_df, hide_index=True)

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.error('🚨 This transaction is likely **Fraudulent**.', icon="🚨")
            st.write(f"**Confidence Score:** {prediction_proba[0][1]:.2%}")
        else:
            st.success('✅ This transaction is likely **Legitimate**.', icon="✅")
            st.write(f"**Confidence Score:** {prediction_proba[0][0]:.2%}")