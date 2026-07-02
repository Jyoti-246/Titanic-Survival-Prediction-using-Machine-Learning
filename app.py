import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger survived using Machine Learning.")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "EDA",
        "Prediction",
        "Model Comparison"
    ]
)

if page == "Home":

    st.header("About Project")

    st.write("""
    This project predicts whether a Titanic passenger survived.

    Algorithms Used:

    - Logistic Regression
    - KNN
    - SVM
    - Kernel SVM
    - Naive Bayes
    - Decision Tree
    - Random Forest
    """)

elif page == "EDA":

    st.header("📊 Exploratory Data Analysis")

    df = pd.read_csv("dataset/train.csv")
    st.subheader("Dataset Preview")

    st.dataframe(df.head())
    st.subheader("Dataset Shape")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())
    st.subheader("Survival Count")

    st.bar_chart(df["Survived"].value_counts())
    st.subheader("Survival by Gender")

    
    gender = df.groupby("Sex")["Survived"].mean()

    
    st.bar_chart(gender)    
    st.subheader("Survival by Passenger Class")

    pclass = df.groupby("Pclass")["Survived"].mean()

    st.bar_chart(pclass)    
    st.subheader("Age Distribution")

    st.bar_chart(df["Age"].dropna())
    st.subheader("Fare Distribution")

    st.bar_chart(df["Fare"])

elif page == "Prediction":

    st.header("Prediction")

    algorithm = st.selectbox(
        "Select Algorithm",
        [
            "Logistic Regression",
            "KNN",
            "SVM",
            "Kernel SVM",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest"
        ]
    )

    st.write("Selected:", algorithm)

    if algorithm == "Logistic Regression":
        model = joblib.load("models/logistic_regression.pkl")

    elif algorithm == "KNN":
        model = joblib.load("models/knn.pkl")

    elif algorithm == "SVM":
        model = joblib.load("models/svm.pkl")

    elif algorithm == "Kernel SVM":
        model = joblib.load("models/kernel_svm.pkl")

    elif algorithm == "Naive Bayes":
        model = joblib.load("models/naive_bayes.pkl")

    elif algorithm == "Decision Tree":
        model = joblib.load("models/Decision_tree_classification.pkl")

    else:
        model = joblib.load("models/random_forest.pkl")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])

    sex = st.selectbox("Sex", ["male", "female"])

    age = st.number_input("Age", min_value=0, max_value=100, value=25)

    sibsp = st.number_input("Siblings/Spouses", min_value=0, value=0)

    parch = st.number_input("Parents/Children", min_value=0, value=0)

    fare = st.number_input("Fare", min_value=0.0, value=50.0)

    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

    encoder = joblib.load("models/encoder.pkl")

    features = np.array([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    ]], dtype=object)

    features = encoder.transform(features)

    if algorithm in ["Logistic Regression", "KNN", "SVM", "Kernel SVM"]:
        scaler = joblib.load("models/scaler.pkl")
        features = scaler.transform(features)

    # Predict
    if st.button("Predict"):

        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ Passenger Survived")
        else:
            st.error("❌ Passenger Did Not Survive")

elif page == "Model Comparison":

    import pandas as pd

    st.header("📊 Model Comparison")

    data = {
        "Algorithm": [
            "Logistic Regression",
            "KNN",
            "SVM",
            "Kernel SVM",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            0.8045,
            0.8212,
            0.8101,
            0.8101,
            0.7989,
            0.7654,
            0.8436
        ],
        "Precision": [
            0.7500,
            0.7937,
            0.7966,
            0.7966,
            0.7324,
            0.7077,
            0.8361
        ],
        "Recall": [
            0.7391,
            0.7246,
            0.6812,
            0.6812,
            0.7536,
            0.6667,
            0.7391
        ],
        "F1 Score": [
            0.7445,
            0.7576,
            0.7344,
            0.7344,
            0.7429,
            0.6866,
            0.7846
        ]
    }

    df = pd.DataFrame(data)

    display_df = df.copy()

    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

    st.dataframe(display_df, use_container_width=True)    

    st.subheader("📈 Accuracy Comparison")
    st.bar_chart(df.set_index("Algorithm")["Accuracy"])

    st.success("🏆 Best Model: Random Forest")

    st.write("""
    **Conclusion**

    Seven machine learning classification algorithms were evaluated on the
    Titanic dataset. Random Forest achieved the highest accuracy (84.36%)
    and the highest F1-score (78.46%), making it the best-performing model.
    Therefore, Random Forest was selected as the final model for deployment.
    """)

