import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import base64
import datetime
import matplotlib.pyplot as plt
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #111;
}

section[data-testid="stSidebar"] * {
    color: white;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)
# ---------- Load Model ----------
model = joblib.load("model/house_price_pipeline.pkl")
# ---------- Function to Set Background ----------
import os
import base64

def set_bg(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    ext = os.path.splitext(image_path)[1][1:]  # gets png/jpg/jpeg

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/{ext};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- Sidebar Navigation ----------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Predict Price", "About", "Admin Panel"]
)

# ================= HOME =================
if page == "Home":

    set_bg("images/home_bg2.png")
    

    st.markdown("""
    <style>
    .overlay-box {
        background: rgba(0, 0, 0, 0.55);
        padding: 40px;
        border-radius: 10px;
        max-width: 700px;
        margin: auto;
        color: white;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    .text {
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="overlay-box">

    <div class="title">House Price Prediction System</div>

    <div class="subtitle">
    Estimate property value using machine learning based on key housing features.
    </div>

    <div class="section-title">Key Features</div>
    <div class="text">
    • Instant price estimation based on property details<br>
    • Simple and user-friendly interface<br>
    • Supports major housing attributes used in valuation<br>
    </div>

    <div class="section-title">Technology</div>
    <div class="text">
    • Random Forest Regression Model<br>
    • Python & Scikit-learn<br>
    • Streamlit Web Application<br>
    </div>

    <div class="section-title">How to Use</div>
    <div class="text">
    Use the navigation panel to enter property details and obtain an estimated price.
    </div>

    </div>
    """, unsafe_allow_html=True)

# ================= PREDICTION PAGE =================
elif page == "Predict Price":

    set_bg("images/predict_bg3.png")

    st.markdown("""
    <h1 style='text-align:center; color:white;
    text-shadow:2px 2px 6px black;'>
    Predict House Price
    </h1>
    """, unsafe_allow_html=True)

    # -------- Row 1 --------
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Area (sq.ft)",min_value=1,step=1)
        #if area < 1650 or area > 16200:
         #   st.warning("Typical dataset range: 1650 – 16200 sq.ft, prediction may be unrealistic.")
        bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, step=1)

    with col2:
        stories = st.number_input("Floors", min_value=0, step=1)

        mainroad = st.selectbox("Near Main Road", ["Yes", "No"])
        mainrd_val = 1 if mainroad == "Yes" else 0

        g_room = st.selectbox("Guestroom Available", ["Yes", "No"])
        guest_val = 1 if g_room == "Yes" else 0

    with col3:
        basement = st.selectbox("Basement Available", ["Yes", "No"])
        base_val = 1 if basement == "Yes" else 0

        hot_water = st.selectbox("Hot Water Heating", ["Yes", "No"])
        water_val = 1 if hot_water == "Yes" else 0

        ac = st.selectbox("Air Conditioning", ["Yes", "No"])
        ac_val = 1 if ac == "Yes" else 0

    # -------- Row 2 --------
    col4, col5, col6 = st.columns(3)

    with col4:
        parking = st.number_input("Parking Spaces", min_value=0, step=1)

    with col5:
        pref_area = st.selectbox("Prime Location", ["Yes", "No"])
        parea_val = 1 if pref_area == "Yes" else 0

    with col6:
        furn_status = st.selectbox(
            "Furnishing Status",
            ["unfurnished", "semi-furnished", "furnished"]
        )

        # Convert to numeric
        if furn_status == "unfurnished":
            furn_val = 0
        elif furn_status == "semi-furnished":
            furn_val = 1
        else:
            furn_val = 2

    # -------- Prediction --------
    if st.button("Predict Price"):

        input_data = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad": [mainrd_val],
            "guestroom": [guest_val],
            "basement": [base_val],   # FIXED
            "hotwaterheating": [water_val],
            "airconditioning": [ac_val],
            "parking": [parking],
            "prefarea": [parea_val],
            "furnishingstatus": [furn_val]  # FIXED
        })

        prediction = model.predict(input_data)[0]
        price=prediction/100000
        st.markdown(f"""
        <div style="
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        margin-top: 10px;
        ">
        Estimated Price (Rs.): ₹ {prediction:,.0f} <br>
        Estimated Price (Lakhs): ₹ {price:,.2f} Lakhs
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Feature Importance")
        rf_model=model.named_steps['model']
        importances=rf_model.feature_importances_
        columns=joblib.load("model/column.pkl")
        feature_names=columns

        fig, ax=plt.subplots(figsize=(8,5))
        ax.barh(feature_names,importances)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature importance from Random Forest")
        st.pyplot(fig)
        
        # log save
        log_file = "logs/user_logs.csv"

# Create folder if not exists
        os.makedirs("logs", exist_ok=True)

# Prepare log entry
        log_entry = pd.DataFrame([{
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainrd_val,
        "guestroom": guest_val,
        "basement": base_val,   # FIXED
        "hotwaterheating": water_val,
        "airconditioning": ac_val,
        "parking": parking,
        "prefarea": parea_val,
        "furnishingstatus": furn_val,
        "Predicted Price": price
        }])

# Save log
        if os.path.exists(log_file):
          log_entry.to_csv(log_file, mode='a', header=False, index=False)
        else:
          log_entry.to_csv(log_file, index=False)
#        st.success(f"Estimated Price: ₹ {prediction:,.0f}")

# ================= ABOUT =================
elif page == "About":
    set_bg("images/about_bg.png")
    st.markdown("""<style>
    .overlay-box {
        background: rgba(0, 0, 0, 0.55);
        padding: 40px;
        border-radius: 10px;
        max-width: 700px;
        color: white;
    }

    .subtitle {
        font-size: 24px;
        font-weight:650;
        margin-bottom: 25px;
    }

    .section {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
    }

    .text {
        font-size: 16px;
        line-height: 1.6;
    }
    </style>""",unsafe_allow_html=True)
    st.markdown("""
    <div class="overlay-box">
                
       <div class="subtitle">This project demonstrates how Machine Learning can estimate property prices
    based on fundamental housing features.</div>
                
    <div class="section"> Objective</div>
      <div class="text">To build a simple system that predicts house prices using historical data. <br></div>
    
    <div class="section"> How It Works</div>
      <div class="text"> 1. User enters property details  <br>
        2. Model processes inputs  <br>
        3. Predicted price is displayed instantly <br> </div>
    
    <div class="section"> Model Used</div>
      <div class="text">Random Forest Regressor — a powerful algorithm for tabular data. <br></div>
    
    <div class="section"> Tools & Technologies</div>
      <div class="text"> Python Pandas & NumPy  <br>
         - Scikit-learn <br> 
         - Streamlit  </div>
    </div>
    """, unsafe_allow_html=True)

# ================= ADMIN PANEL =================
elif page == "Admin Panel":

    st.title("🔐 Admin Panel")

    password = st.text_input("Enter Admin Password", type="password")

    if password == "admin123":   # Change password here

        st.success("Access Granted")
        log_file = "logs/user_logs.csv"
        if os.path.exists("logs/user_logs.csv"):
            logs = pd.read_csv("logs/user_logs.csv")
            st.subheader("📊 Prediction Logs")
            st.dataframe(logs)
        else:
            st.warning("No logs available yet.")

    elif password != "":
        st.error("Incorrect Password")