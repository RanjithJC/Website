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
st.set_page_config(
    page_title="Personnel Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CUSTOM CSS FOR BETTER LOOK
# ----------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    /* Cards */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .stat-number {
        font-size: 36px;
        font-weight: bold;
        color: #2980b9;
    }
    .metric-label {
        font-size: 16px;
        color: #7f8c8d;
    }
    /* Buttons */
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1f618d;
        color: white;
    }
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    /* Sidebar headers */
    .sidebar-header {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
# LOAD DATA & MODEL (FIXED FILES)
# ----------------------------
@st.cache_resource
def load_data_and_model():
    # Load CSV
    csv_path = "final_dataset.csv"
    if not os.path.exists(csv_path):
        st.error(f"❌ CSV file '{csv_path}' not found. Please place it in the same directory.")
        return None, None, [], None, {}

    df = pd.read_csv(csv_path)

    # Load model
    model_path = "stress_risk_metadata_v3.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please place it in the same directory.")
        return df, None, [], None, {}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Auto-detect target (last column) and features
    all_cols = df.columns.tolist()
    target = all_cols[-1]   # assume last column is target
    features = [col for col in all_cols if col != target]

    # Encode categorical features
    label_encoders = {}
    for col in features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, model, features, target, label_encoders

# ----------------------------
# LOGIN
# ----------------------------
def login():
    st.title("🔐 Personnel Analytics")
    st.markdown("<p style='font-size:18px;'>Please log in to access the system</p>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            # Hardcoded credentials (replace with your own)
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

# ----------------------------
# PAGE: DASHBOARD
# ----------------------------
def dashboard():
    st.title("📊 Dashboard")
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Data not loaded. Check that 'personnel_data.csv' exists.")
        return

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Total Personnel</div>
            <div class="stat-number">{df.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Features</div>
            <div class="stat-number">{df.shape[1]-1}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        target = st.session_state.target_column
        if target and target in df.columns:
            avg = df[target].mean()
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Avg {target}</div>
                <div class="stat-number">{avg:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Visualizations
    st.subheader("📈 Distributions")
    cols = st.multiselect("Select columns to visualize", df.columns, default=df.columns[:2])
    if cols:
        for col in cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}", color_discrete_sequence=['#2980b9'])
            fig.update_layout(bargap=0.1)
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
        st.warning("⚠️ Data or model not loaded. Check your files.")
        return

    st.markdown("### Select an existing person")
    df_display = df.copy()
    df_display['Person ID'] = df_display.index
    person_id = st.selectbox("Choose a person", df_display.index.tolist(), format_func=lambda x: f"Person {x}")

    if st.button("🔮 Predict Score", use_container_width=True):
        row = df.loc[person_id, features].values.reshape(1, -1)
        pred = model.predict(row)[0]
        st.success(f"✅ Predicted {target}: **{pred:.2f}**")

    st.markdown("---")
    st.markdown("### Or enter new values manually")
    with st.form("manual_prediction"):
        inputs = {}
        for col in features:
            # if categorical, show selectbox with original values (but we only have encoded)
            # For simplicity, we'll use number inputs
            inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))
        submitted = st.form_submit_button("🔮 Predict")
        if submitted:
            input_array = np.array([inputs[col] for col in features]).reshape(1, -1)
            pred = model.predict(input_array)[0]
            st.success(f"✅ Predicted {target}: **{pred:.2f}**")

# ----------------------------
# PAGE: ADD PERSONNEL
# ----------------------------
def add_personnel():
    st.title("➕ Add New Personnel")
    df = st.session_state.df
    features = st.session_state.feature_columns
    target = st.session_state.target_column
    model = st.session_state.model

    if df is None:
        st.warning("⚠️ Data not loaded. Cannot add personnel.")
        return

    with st.form("add_person_form"):
        st.markdown("### Enter details")
        new_data = {}
        cols = st.columns(2)
        for i, col in enumerate(features):
            with cols[i % 2]:
                if col in st.session_state.label_encoders:
                    # show a text input; we'll handle encoding later
                    new_data[col] = st.text_input(f"{col}", placeholder="Enter value")
                else:
                    new_data[col] = st.number_input(f"{col}", value=0.0, step=0.1)

        target_val = st.number_input(f"{target} (optional, leave blank to predict)", value=None, step=0.1)
        submitted = st.form_submit_button("➕ Add Personnel", use_container_width=True)

        if submitted:
            # Build new row
            new_row = {}
            for col in features:
                val = new_data[col]
                if col in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[col]
                    # if value is empty, use most frequent
                    if val == "" or val is None:
                        val = le.classes_[0]
                        st.info(f"Using default '{val}' for {col}")
                    else:
                        try:
                            val = le.transform([str(val)])[0]
                        except:
                            val = le.classes_[0]
                            st.warning(f"Value not seen, using default '{le.classes_[0]}' for {col}")
                    new_row[col] = val
                else:
                    new_row[col] = val

            if target_val is not None and target_val != "":
                new_row[target] = target_val
            else:
                if model is not None:
                    input_array = np.array([new_row[col] for col in features]).reshape(1, -1)
                    pred = model.predict(input_array)[0]
                    new_row[target] = pred
                    st.info(f"Predicted {target}: {pred:.2f}")
                else:
                    new_row[target] = 0.0

            # Append
            df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df = df_new
            # Save back to CSV
            df_new.to_csv("personnel_data.csv", index=False)
            st.success("✅ Personnel added successfully!")
            st.rerun()

# ----------------------------
# MAIN NAVIGATION
# ----------------------------
def main():
    if not st.session_state.logged_in:
        login()
        return

    # Load data (cached)
    if st.session_state.df is None:
        df, model, features, target, le = load_data_and_model()
        if df is not None:
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.feature_columns = features
            st.session_state.target_column = target
            st.session_state.label_encoders = le

    # Sidebar navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/user-group-man-man.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Personnel Score", "Add Personnel"], index=0)

    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
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