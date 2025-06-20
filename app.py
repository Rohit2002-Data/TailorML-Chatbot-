# TailorML - Enhanced with Gemini AI Performance-Aware Assistant
import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import r2_score, classification_report
from fpdf import FPDF
import time
import google.generativeai as genai

# ======== Configure Gemini AI ========
genai.configure(api_key="YOUR_API_KEY")  # Replace with your Gemini API Key
model = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="TailorML", page_icon="🧠", layout="centered")

# ======== Styling ========
st.markdown("""
<style>
.chat-input-container label {display: none;}
[data-testid="stSidebar"] {background-color: #F5F7FA;}
[data-testid="stAppViewContainer"] > .main {background-color: #FDFDFD;}
.stButton > button {border-radius: 1rem; padding: 0.5rem 1.2rem; background-color: #4F46E5; color: white;}
</style>
""", unsafe_allow_html=True)

# ======== Missing Value Handling ========
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        missing_ratio = df[col].isnull().mean()
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            strategy = 'mean' if missing_ratio <= 0.3 else 'median'
            imputer = SimpleImputer(strategy=strategy)
        df[col] = imputer.fit_transform(df[[col]])
    return df

# ======== Model Selector ========
def get_models(task_type):
    if task_type == 'classification':
        return {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier()
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }

# ======== App State Initialization ========
st.title(":brain: TailorML: Your Predictive Chat Assistant")
if "stage" not in st.session_state:
    st.session_state.stage = "start"
    st.session_state.df = None

# ======== Upload Stage ========
if st.session_state.stage == "start":
    st.chat_message("ai").write("\U0001f44b Hi, I’m **TailorML**! Let’s predict together. Start by uploading your dataset \U0001f4c2")
    file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = handle_missing_values(df)
        st.session_state.df = df
        st.session_state.stage = "uploaded"
        st.rerun()

# ======== Dataset Preview Stage ========
elif st.session_state.stage == "uploaded":
    st.chat_message("ai").write("\U0001f440 Here's a preview of your dataset:")
    st.dataframe(st.session_state.df.head())
    st.chat_message("ai").write(f"\U0001f4d0 Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

    if st.button("✅ All Good, Next"):
        st.session_state.stage = "choose_target"
        st.rerun()

# ======== Target Selection Stage ========
elif st.session_state.stage == "choose_target":
    st.chat_message("ai").write("🎯 What would you like to predict? Choose a target column:")
    target = st.selectbox("Select the target column", st.session_state.df.columns)
    if target:
        st.session_state.target = target
        if st.button("🔍 Confirm Target"):
            st.session_state.stage = "predict"
            st.rerun()

# ======== Model Training and Prediction Stage ========
elif st.session_state.stage == "predict":
    st.chat_message("ai").write("⚙️ Training ML models on your data...")
    time.sleep(1)

    df = st.session_state.df
    target = st.session_state.target
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    task_type = "classification" if y.dtype == 'object' or y.nunique() <= 10 else "regression"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models = get_models(task_type)

    best_model = None
    best_score = -np.inf
    best_y_pred = None
    best_model_obj = None
    results = {}

    progress = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred) if task_type == "regression" else classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
            results[name] = round(score, 4)
            if score > best_score:
                best_score = score
                best_model = name
                best_model_obj = model
                best_y_pred = y_pred
        except Exception as e:
            results[name] = f"❌ {str(e)}"
        progress.progress((i + 1) / len(models))

    st.session_state.update({
        "stage": "results",
        "results": results,
        "best_model": best_model,
        "best_score": best_score,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": best_y_pred,
        "best_model_obj": best_model_obj,
        "X_train": X_train,
        "task_type": task_type
    })
    st.rerun()

# ======== Results & Insights Stage ========
elif st.session_state.stage == "results":
    st.chat_message("ai").write(f"🏅 Best model: **{st.session_state.best_model}** | Score: **{st.session_state.best_score:.2f}**")
    st.chat_message("ai").write("📊 All model performances:")
    st.dataframe(pd.DataFrame.from_dict(st.session_state.results, orient='index', columns=['Score']))

    # ======== Gemini Assistant with ML Context Awareness ========
    user_input = st.chat_input("💬 Ask TailorML about models, performance, or concepts...")
    if user_input:
        st.chat_message("user").write(user_input)
        model_perf = "\n".join([f"{k}: {v}" for k, v in st.session_state.results.items()]) if "results" in st.session_state else "No model scores available."
        context = f"""
        You are TailorML, a smart machine learning assistant.

        Dataset preview: {st.session_state.df.head(3).to_dict() if 'df' in st.session_state else 'Not loaded'}
        Target: {st.session_state.get('target', 'N/A')}
        Task: {st.session_state.get('task_type', 'Unknown')}
        Best model: {st.session_state.get('best_model', 'N/A')} with score {st.session_state.get('best_score', 'N/A')}
        Model performances:
        {model_perf}

        Respond clearly using this context if user asks about:
        - model comparison
        - predictions
        - ML explanation (e.g., SHAP, LIME, F1-score)
        - Why a model performed well

        If user asks generic ML questions (like "What is overfitting?"), explain them too.
        Now answer this:
        '''{user_input}'''
        """
        try:
            with st.spinner("TailorML is thinking... \U0001f914"):
                response = model.generate_content(context)
                st.chat_message("ai").write(response.text)
        except Exception as e:
            st.chat_message("ai").write(f"⚠️ Gemini couldn't respond: {e}")
