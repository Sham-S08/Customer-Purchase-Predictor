import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------- #
# 🎨 PAGE CONFIGURATION
# --------------------------- #
st.set_page_config(page_title="🛍️ Smart Purchase Predictor", layout="centered")

# --------------------------- #
# 🌈 CUSTOM CSS
# --------------------------- #
st.markdown("""
    <style>
        body { background-color: #f8fafc; }
        h1, h2, h3, h4, h5, h6 { color: #2b2d42; font-family: 'Poppins', sans-serif; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 50px;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #43a047;
            transform: scale(1.05);
        }
        .stNumberInput, .stTextInput, .stSelectbox, .stSlider {
            margin-bottom: 25px !important;
        }
        .stCaption {
            margin-top: -8px !important;
            margin-bottom: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------- #
# 📊 LOAD DATA & MODEL
# --------------------------- #
dataset = pd.read_csv("Data/online_shoppers_intention.csv")
model = joblib.load("src/purchase_model.pkl")
scaler = joblib.load("src/scaler.pkl")

feature_cols = [
    'Administrative', 'Administrative_Duration', 'Informational',
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
    'OperatingSystems', 'Browser', 'Region', 'TrafficType',
    'VisitorType', 'Weekend'
]

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
dataset['Month'] = dataset['Month'].map(month_map)
dataset['VisitorType'] = dataset['VisitorType'].apply(lambda x: 1 if x == "Returning_Visitor" else 0)
dataset['Weekend'] = dataset['Weekend'].apply(lambda x: 1 if x else 0)

# --------------------------- #
# ⚙️ SESSION STATE
# --------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []
if "count" not in st.session_state:
    st.session_state.count = 0

# --------------------------- #
# 🧭 SIDEBAR
# --------------------------- #
st.sidebar.header("🧭 Navigation")
st.sidebar.markdown("Use this panel to view info and history.")

st.sidebar.markdown("**Model:** Random Forest Classifier")
st.sidebar.markdown("[📘 Dataset Source](https://archive.ics.uci.edu/ml/datasets/online+shoppers+intention)")

if st.session_state.history:
    st.sidebar.subheader("📜 Previous Predictions")
    for idx, record in enumerate(st.session_state.history[-5:][::-1]):
        st.sidebar.write(f"• {record['label']} — {record['confidence']:.1f}%")

# --------------------------- #
# 👋 GREETING
# --------------------------- #
st.title("🛍️ Smart Purchase Predictor")
st.success("Welcome! Let’s analyze how browsing behavior affects purchase likelihood. 🚀")

# --------------------------- #
# 🧩 TABS LAYOUT
# --------------------------- #
tab1, tab2, tab3 = st.tabs(["📖 Overview", "🔮 Prediction", "💡 Insights"])

# --------------------------- #
# 📖 TAB 1: OVERVIEW
# --------------------------- #
with tab1:
    st.header("📖 App Overview")
    st.write("""
    This app predicts whether an online shopper is **likely to make a purchase** based on browsing behavior metrics.
    
    You only need to enter **4 simple parameters**, and the model will instantly estimate the **purchase probability**.
    """)

    with st.expander("🧠 Learn More About the Model"):
        st.markdown("""
        - **Algorithm Used:** Random Forest Classifier  
        - **Dataset:** Online Shoppers Intention Dataset (UCI)  
        - **Purpose:** To predict the 'Revenue' (purchase or not)  
        - **Trained Features:** 17 behavioral and technical attributes  
        """)

    with st.expander("💬 Why This Matters"):
        st.markdown("""
        Understanding user intent helps businesses:
        - Optimize page design 🧩  
        - Identify high-value visitors 💰  
        - Improve conversion rate 📈  
        """)

# --------------------------- #
# 🔮 TAB 2: PREDICTION
# --------------------------- #
with tab2:
    st.header("🔮 Make a Prediction")
    st.write("Provide a few simple details about the browsing session below:")

    # User Inputs
    time_on_product_pages_min = st.number_input(
        "🕒 Average Time on Product Pages (minutes)",
        min_value=0.0, value=3.0,
        help="Average duration users spend exploring product pages. Longer time means higher interest."
    )
    pages_with_value = st.number_input(
        "💰 Average Page Engagement Value",
        min_value=0.0, value=10.0,
        help="How valuable each visited page is in terms of conversions."
    )
    quick_leave_rate = st.number_input(
        "🚪 Quick Leave Percentage (Bounce Rate, 0.0 - 1.0)",
        min_value=0.0, max_value=1.0, value=0.02,
        help="Shows how many visitors leave after one page. Lower means more exploration."
    )
    exit_rate = st.number_input(
        "↩️ Site Exit Likelihood (Exit Rate, 0.0 - 1.0)",
        min_value=0.0, max_value=1.0, value=0.05,
        help="Percentage of users leaving after a page. High values mean lost interest."
    )

    # Data Prep
    time_on_product_pages_sec = time_on_product_pages_min * 60
    numeric_cols = dataset[feature_cols].select_dtypes(include=['int64', 'float64']).columns
    input_df = pd.DataFrame([dataset[numeric_cols].mean()])

    # Fill categorical columns
    for col in ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']:
        input_df[col] = dataset[col].mode()[0]

    # Update user input
    input_df['BounceRates'] = quick_leave_rate
    input_df['ExitRates'] = exit_rate
    input_df['PageValues'] = pages_with_value
    input_df['ProductRelated_Duration'] = time_on_product_pages_sec

    input_df = input_df[feature_cols]
    scaled_input = scaler.transform(input_df)

    # Prediction Button
    if st.button("🧠 Predict Now"):
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        result_text = "🟢 Likely to Purchase" if prediction == 1 else "🔴 Unlikely to Purchase"
        st.subheader(result_text)
        st.progress(int(probability * 100))
        st.write(f"**Confidence:** {probability*100:.2f}%")

        # Animation & Tracking
        if prediction == 1:
            st.balloons()
        else:
            st.snow()

        st.session_state.count += 1
        st.session_state.history.append({
            "label": result_text,
            "confidence": probability * 100
        })

# --------------------------- #
# 💡 TAB 3: INSIGHTS
# --------------------------- #
with tab3:
    st.header("💡 Behavioral Insights")
    st.write("Explore what different metrics mean for customer purchase patterns.")

    st.markdown("""
    - 🕒 **Time on Product Pages:** Longer times correlate with higher purchase intent.  
    - 💰 **Page Value:** Reflects likelihood of conversion — the higher, the better.  
    - 🚪 **Bounce Rate:** A high bounce rate signals disinterest or poor page engagement.  
    - ↩️ **Exit Rate:** Helps identify pages that cause users to drop off.  
    """)

    st.info("💬 Tip: Combine this app with Google Analytics data to identify high-potential leads.")

    st.markdown("---")
    st.write("✨ **Total Predictions Made:**", st.session_state.count)
    st.caption("Built with ❤️ using Streamlit + Scikit-learn")

