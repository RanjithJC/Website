import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Personnel Analytics", layout="wide")

# ----------------------------
# SESSION STATE INITIALIZATION
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# ----------------------------
# LOGIN
# ----------------------------
def login():
    st.title("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            # Hardcoded credentials (replace with your own logic)
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")

# ----------------------------
# LOAD DATA & MODEL
# ----------------------------
def load_data():
    st.sidebar.header("📂 Data & Model Upload")
    uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    uploaded_pkl = st.sidebar.file_uploader("Upload Pickle Model", type=["pkl"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.session_state.df = df
        st.sidebar.success("CSV loaded successfully!")
    else:
        # Try to load a local file if exists
        if os.path.exists("personnel_data.csv") and st.session_state.df is None:
            df = pd.read_csv("personnel_data.csv")
            st.session_state.df = df
            st.sidebar.info("Loaded local CSV: personnel_data.csv")

    if uploaded_pkl is not None:
        model = pickle.load(uploaded_pkl)
        st.session_state.model = model
        st.sidebar.success("Model loaded successfully!")
    else:
        if os.path.exists("model.pkl") and st.session_state.model is None:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            st.session_state.model = model
            st.sidebar.info("Loaded local model: model.pkl")

    # If df is loaded, auto-detect features and target
    if st.session_state.df is not None:
        df = st.session_state.df
        all_cols = df.columns.tolist()
        # Assume the last column is the target (or ask user)
        target = st.sidebar.selectbox("Select target column", all_cols, index=len(all_cols)-1)
        st.session_state.target_column = target
        # Features are all other columns
        features = [col for col in all_cols if col != target]
        st.session_state.feature_columns = features
        st.sidebar.write(f"**Features:** {features}")
        st.sidebar.write(f"**Target:** {target}")

        # Encode categorical features for model consistency
        for col in features:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                st.session_state.label_encoders[col] = le
        st.session_state.df = df

# ----------------------------
# PAGE: DASHBOARD
# ----------------------------
def dashboard():
    st.title("📊 Dashboard")
    df = st.session_state.df
    if df is None:
        st.warning("Please upload a CSV file first.")
        return

    st.subheader("Data Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head(10))

    # Basic stats
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Visualizations
    st.subheader("Distribution of Features")
    cols = st.multiselect("Select columns to plot", df.columns, default=df.columns[:2])
    if cols:
        for col in cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

    # Target distribution if exists
    target = st.session_state.target_column
    if target and target in df.columns:
        st.subheader(f"Target Variable: {target}")
        fig = px.histogram(df, x=target, title=f"Distribution of {target}")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# PAGE: PERSONNEL SCORE
# ----------------------------
def personnel_score():
    st.title("🎯 Personnel Score Prediction")
    df = st.session_state.df
    model = st.session_state.model
    features = st.session_state.feature_columns
    target = st.session_state.target_column

    if df is None or model is None:
        st.warning("Please load both CSV and model first.")
        return

    st.subheader("Select a person from the dataset")
    # Add an index column to identify rows
    df_display = df.copy()
    df_display['Person ID'] = df_display.index
    person_id = st.selectbox("Choose a person", df_display.index.tolist(), format_func=lambda x: f"Person {x}")

    if st.button("Predict Score"):
        row = df.loc[person_id, features].values.reshape(1, -1)
        # Handle any missing encoding (it should already be encoded)
        pred = model.predict(row)[0]
        st.success(f"Predicted {target}: **{pred:.2f}**")

    # Manual input for prediction
    st.subheader("Or enter new features manually")
    with st.form("manual_prediction"):
        inputs = {}
        for col in features:
            # If the column was categorical, show a selectbox with unique values
            if col in st.session_state.label_encoders:
                unique_vals = df[col].unique()  # encoded values
                # We need to map back to original strings? Not necessary; we'll use numeric.
                # Better to show original values if we stored them.
                # For simplicity, we use numeric input.
                inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))
            else:
                inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_array = np.array([inputs[col] for col in features]).reshape(1, -1)
            pred = model.predict(input_array)[0]
            st.success(f"Predicted {target}: **{pred:.2f}**")

# ----------------------------
# PAGE: ADD PERSONNEL
# ----------------------------
def add_personnel():
    st.title("➕ Add New Personnel")
    df = st.session_state.df
    features = st.session_state.feature_columns
    target = st.session_state.target_column

    if df is None:
        st.warning("Please upload a CSV file first.")
        return

    st.subheader("Enter details for new personnel")
    with st.form("add_person_form"):
        new_data = {}
        for col in features:
            # Use original column names; if categorical, show selectbox with original strings
            # But we need to encode before saving. We'll store original string, then encode.
            if col in st.session_state.label_encoders:
                # Get original unique values from the original df (before encoding)
                # Since we don't have original, we can use the encoded values and map back?
                # Better to store original values in a separate dict; we'll simplify by using text input.
                new_data[col] = st.text_input(f"{col} (categorical)", value="")
            else:
                new_data[col] = st.number_input(f"{col}", value=0.0)
        # Target can be optional; if not provided, we can predict it.
        target_val = st.number_input(f"{target} (optional, leave blank to predict)", value=None, step=0.1)
        submitted = st.form_submit_button("Add Personnel")

        if submitted:
            # Create a new row
            new_row = {}
            for col in features:
                val = new_data[col]
                if col in st.session_state.label_encoders:
                    # Encode the string input
                    le = st.session_state.label_encoders[col]
                    try:
                        # If the value is not seen, assign a default (e.g., most frequent)
                        if val not in le.classes_:
                            st.warning(f"Value '{val}' not seen in training, using most frequent.")
                            val = le.classes_[0]
                        encoded = le.transform([val])[0]
                    except:
                        encoded = 0
                    new_row[col] = encoded
                else:
                    new_row[col] = val
            # Add target if provided, else predict
            if target_val is not None:
                new_row[target] = target_val
            else:
                # Predict using model if available
                if st.session_state.model is not None:
                    input_array = np.array([new_row[col] for col in features]).reshape(1, -1)
                    pred = st.session_state.model.predict(input_array)[0]
                    new_row[target] = pred
                    st.info(f"Predicted {target}: {pred:.2f}")
                else:
                    new_row[target] = 0.0  # placeholder

            # Append to dataframe
            df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df = df_new
            st.success("New personnel added successfully!")
            # Optionally save to CSV
            df_new.to_csv("personnel_data.csv", index=False)
            st.rerun()

# ----------------------------
# MAIN NAVIGATION
# ----------------------------
def main():
    if not st.session_state.logged_in:
        login()
        return

    # Load data/model (sidebar)
    load_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Personnel Score", "Add Personnel"])

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Page routing
    if page == "Dashboard":
        dashboard()
    elif page == "Personnel Score":
        personnel_score()
    elif page == "Add Personnel":
        add_personnel()

if __name__ == "__main__":
    main()