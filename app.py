import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC



st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """
    Loads the heart disease dataset from a remote URL.
    """
    # This is a common, reliable source for this dataset
    DATA_URL = "./heart.csv"
    try:
        df = pd.read_csv(DATA_URL)
        if 'HeartDisease' not in df.columns:
            st.error("Target column 'HeartDisease' not found in dataset.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series()
        
        # Drop rows with 0 Cholesterol that are problematic (as noted in the notebook)
        # This mirrors a common preprocessing step for this dataset
        df = df[df['Cholesterol'] != 0]

        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        return df, X, y
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

# Load the data
df_raw, X, y = load_data()


categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' 
)


@st.cache_resource
def train_models(X_train, y_train, _preprocessor):

    models = {}

    # 1. Random Forest 
    rf_pipeline = Pipeline(steps=[('preprocessor', _preprocessor),
                                  ('model', RandomForestClassifier(n_estimators=200, random_state=42))])
    rf_pipeline.fit(X_train, y_train)
    models['Random Forest'] = rf_pipeline

    # 2. Support Vector Machine 
    svm_pipeline = Pipeline(steps=[('preprocessor', _preprocessor),
                                   ('model', SVC(probability=True, kernel='rbf'))])
    svm_pipeline.fit(X_train, y_train)
    models['Support Vector Machine (SVM)'] = svm_pipeline

    # 3. Extra Trees
    et_pipeline = Pipeline(steps=[('preprocessor', _preprocessor),
                                    ('model', ExtraTreesClassifier(n_estimators=200, random_state=42))])
    et_pipeline.fit(X_train, y_train)
    models['Extra Trees'] = et_pipeline
    
    return models

# Train models only if data was loaded successfully
if not X.empty:
    models = train_models(X, y, preprocessor)
else:
    st.warning("Cannot train models as data is empty.")
    models = {}


st.sidebar.title("Patient Input Features")
st.sidebar.markdown("Adjust the sliders and select options to get a prediction.")

# Create input widgets in the sidebar
with st.sidebar:
    age = st.slider('Age', int(df_raw['Age'].min()), int(df_raw['Age'].max()), 54)
    sex = st.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.selectbox('Chest Pain Type', ('ASY', 'ATA', 'NAP', 'TA'))
    resting_bp = st.slider('Resting Blood Pressure (mm Hg)', int(df_raw['RestingBP'].min()), int(df_raw['RestingBP'].max()), 130)
    cholesterol = st.slider('Cholesterol (mm/dl)', int(df_raw['Cholesterol'].min()), int(df_raw['Cholesterol'].max()), 240)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    resting_ecg = st.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.slider('Maximum Heart Rate', int(df_raw['MaxHR'].min()), int(df_raw['MaxHR'].max()), 150)
    exercise_angina = st.selectbox('Exercise Angina', ('N', 'Y'), format_func=lambda x: 'Yes' if x == 'Y' else 'No')
    oldpeak = st.slider('Oldpeak (ST depression)', float(df_raw['Oldpeak'].min()), float(df_raw['Oldpeak'].max()), 1.0, 0.1)
    st_slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))


st.title("==Heart Disease Prediction==")

input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Selection")
    model_choice = st.selectbox("Select Model for Prediction:", list(models.keys()))
    predict_button = st.button("Predict", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Result")
    
    if predict_button and models:

        model = models[model_choice]
        
       
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
           
            if prediction == 1:
                st.error(f"**Result: High Risk of Heart Disease**", icon="💔")
                st.metric(label="Model Confidence", value=f"{prediction_proba[1]*100:.2f}%")
            else:
                st.success(f"**Result: Low Risk of Heart Disease**", icon="❤️")
                st.metric(label="Model Confidence", value=f"{prediction_proba[0]*100:.2f}%")
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.info("Click 'Predict' to see the result.")

st.divider()


with st.expander("Show Patient Input Data"):
    st.dataframe(input_data.T.rename(columns={0: 'Value'}))


with st.expander("Show Snippet of Training Data"):
    st.dataframe(df_raw.head())