import streamlit as st

st.set_page_config(
    page_title="Soil Fertility Analysis", 
    page_icon="🌱", 
    layout="wide"
)

st.title("🌱 Soil Fertility Analysis Dashboard")
st.markdown("*Data Mining Project for Soil Parameter Classification*")

# Project Overview
st.header("📊 Project Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 What This Project Does")
    st.markdown("""
    A data mining project that analyzes soil samples to predict fertility levels. 
    Uses machine learning classification algorithms to process 13 different soil 
    parameters and determine soil fertility status.
    """)
    
    st.subheader("📋 Soil Parameters")
    st.markdown("""
    **Primary Nutrients:**
    - N (Nitrogen), P (Phosphorus), K (Potassium)
    
    **Chemical Properties:**
    - pH, EC (Electrical Conductivity), OC (Organic Carbon)
    
    **Additional Elements:**
    - S (Sulfur), Zn (Zinc), Fe (Iron), Cu (Copper)
    - Mn (Manganese), B (Boron), OM (Organic Matter)
    """)

with col2:
    st.subheader("🤖 Machine Learning Models")
    st.markdown("""
    - **Decision Tree**: Classification based on soil parameter rules
    - **Random Forest**: Ensemble method for better predictions
    - **KNN**: K-Nearest Neighbors classification
    - **Apriori**: Association rule mining implementation
    """)
    
    st.subheader("🔧 Technical Components")
    st.markdown("""
    - **Data Ingestion**: CSV/Excel file processing
    - **Data Transformation**: Preprocessing and feature engineering
    - **Model Training**: Automated training with parameter tuning
    - **Prediction Pipeline**: Real-time fertility prediction
    - **Metrics**: Model evaluation and performance tracking
    """)

# Project Structure
st.header("📁 Project Structure")

structure_col1, structure_col2 = st.columns(2)

with structure_col1:
    st.markdown("""
    **Core Components:**
    - `data_ingestion.py` - Load and split soil data
    - `model_trainer.py` - Train classification models
    - `predict_pipeline.py` - Make fertility predictions
    - `metrics.py` - Evaluate model performance
    """)

with structure_col2:
    st.markdown("""
    **Model Implementations:**
    - `decision_tree.py` - Decision tree classifier
    - `random_forest.py` - Random forest ensemble
    - `KNN.py` - K-nearest neighbors
    - `apriori.py` - Association rule mining
    """)

# How It Works
st.header("⚙️ How It Works")

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown("""
    **1. Data Input**
    - Load soil analysis data
    - Support for CSV/Excel formats
    - Automatic data validation
    - Train/test data splitting
    """)

with step_col2:
    st.markdown("""
    **2. Model Training**
    - Train multiple algorithms
    - Compare model performance
    - Save trained models
    - Evaluate accuracy metrics
    """)

with step_col3:
    st.markdown("""
    **3. Prediction**
    - Input soil parameter values
    - Get fertility classification
    - View prediction confidence
    - Analyze soil composition
    """)

# Features
st.header("✨ Key Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    **Data Processing:**
    - Multiple file format support
    - Data cleaning and preprocessing
    - Feature transformation pipeline
    - Automated data splitting
    """)

with feature_col2:
    st.markdown("""
    **Machine Learning:**
    - Multiple classification algorithms
    - Model comparison and evaluation
    - Hyperparameter configuration via YAML
    - Prediction pipeline for new samples
    """)

# Technical Details
st.header("💻 Technical Implementation")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    **Dependencies:**
    - pandas, scikit-learn
    - matplotlib, seaborn, plotly
    - streamlit, pydantic
    - pyyaml for configuration
    """)

with tech_col2:
    st.markdown("""
    **Code Quality:**
    - Type checking with mypy
    - Code formatting with black
    - Linting with ruff
    - Testing with pytest
    """)

with tech_col3:
    st.markdown("""
    **Structure:**
    - Modular component design
    - Configuration-driven models
    - Logging system
    - Utils for file operations
    """)

st.sidebar.success("Select a page to begin soil analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Technologies:</strong> Python • Pandas • Scikit-learn • Streamlit</p>
    <p><em>A data mining approach to soil fertility classification</em></p>
</div>
""", unsafe_allow_html=True)