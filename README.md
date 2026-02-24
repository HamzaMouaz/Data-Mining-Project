<<<<<<< HEAD
<<<<<<< HEAD
# 🌱 Soil Fertility Prediction - Data Mining Project

A comprehensive data mining project that predicts soil fertility levels using machine learning classification algorithms. This system analyzes 13 key soil parameters to determine fertility status, helping agricultural professionals make informed decisions about soil management.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Soil Parameters](#soil-parameters)
- [Machine Learning Models](#machine-learning-models)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Components](#project-components)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete data mining pipeline for soil fertility classification. Using supervised learning algorithms, it processes soil analysis data containing 13 different parameters and predicts whether soil samples have high, medium, or low fertility levels.

**Key Achievements:**
- ✅ Multi-model classification system with 4 different algorithms
- ✅ Complete data processing pipeline from ingestion to prediction
- ✅ Web-based dashboard for interactive soil analysis
- ✅ Professional code structure with type checking and testing
- ✅ Configurable model parameters via YAML files

## 📁 Repository Structure

```
DataMining-project/
├── 📄 README.md                    # Project documentation
├── 📄 gui.py                       # Main Streamlit dashboard
├── 📄 test.py                      # Basic test runner
├── 📄 makefile                     # Build automation
├── 📄 pyproject.toml               # Poetry dependencies
├── 📄 poetry.lock                  # Locked dependencies
├── 📄 models_params.yaml           # ML model hyperparameters
├── 📄 mypy.ini                     # Type checking configuration
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 soil_fertility/              # Main package
│   ├── 📄 __init__.py
│   ├── 📄 logger.py                # Logging configuration
│   ├── 📄 utils.py                 # Utility functions
│   │
│   ├── 📂 components/              # Core components
│   │   ├── 📄 data_ingestion.py    # Data loading and splitting
│   │   ├── 📄 model_trainer.py     # Model training orchestration
│   │   ├── 📄 model_utils.py       # Model evaluation utilities
│   │   ├── 📄 metrics.py           # Performance metrics
│   │   ├── 📄 path_config.py       # File path management
│   │   ├── 📄 utils.py             # Component utilities
│   │   │
│   │   ├── 📂 data_transformation/ # Data preprocessing
│   │   │   └── data processing modules
│   │   │
│   │   └── 📂 models/              # ML model implementations
│   │       ├── 📄 decision_tree.py # Decision Tree classifier
│   │       ├── 📄 random_forest.py # Random Forest ensemble
│   │       ├── 📄 KNN.py           # K-Nearest Neighbors
│   │       └── 📄 apriori.py       # Apriori algorithm
│   │
│   └── 📂 pipeline/                # ML pipelines
│       ├── 📄 predict_pipeline.py  # Prediction pipeline
│       └── 📄 train_pipeline.py    # Training pipeline
│
├── 📂 notebook/                    # Jupyter notebooks
│   └── analysis and experimentation notebooks
│
└── 📂 pages/                       # Streamlit pages
    └── additional dashboard pages
```

## ✨ Features

### 🔍 Data Processing
- **Multi-format Support**: Load data from CSV and Excel files
- **Automated Preprocessing**: Data cleaning and feature engineering
- **Smart Splitting**: Automatic train/test data separation (80/20)
- **Data Validation**: Type checking with Pydantic models

### 🤖 Machine Learning
- **Multiple Algorithms**: 4 different classification models
- **Model Comparison**: Automated evaluation and performance metrics
- **Hyperparameter Tuning**: GridSearch optimization support
- **Model Persistence**: Save and load trained models

### 🌐 Web Interface
- **Interactive Dashboard**: Streamlit-based user interface
- **Real-time Predictions**: Input soil parameters and get instant results
- **Visualization**: Charts and graphs for data analysis
- **User-friendly**: Intuitive interface for non-technical users

### 🛠️ Development Tools
- **Code Quality**: Black formatting, Ruff linting, MyPy type checking
- **Testing**: Pytest framework for unit tests
- **Dependency Management**: Poetry for package management
- **Configuration**: YAML-based parameter configuration

## 🧪 Soil Parameters

The system analyzes 13 critical soil parameters:

### Primary Nutrients
- **N (Nitrogen)**: Essential for plant growth and chlorophyll production
- **P (Phosphorus)**: Critical for root development and flowering
- **K (Potassium)**: Important for water regulation and disease resistance

### Chemical Properties
- **pH**: Soil acidity/alkalinity level affecting nutrient availability
- **EC (Electrical Conductivity)**: Measures soil salinity
- **OC (Organic Carbon)**: Indicates organic matter content

### Secondary Elements
- **S (Sulfur)**: Important for protein synthesis
- **OM (Organic Matter)**: Overall soil health indicator

### Micronutrients
- **Zn (Zinc)**: Enzyme activation and growth regulation
- **Fe (Iron)**: Chlorophyll synthesis and electron transport
- **Cu (Copper)**: Enzyme systems and lignin synthesis
- **Mn (Manganese)**: Photosynthesis and nitrogen metabolism
- **B (Boron)**: Cell wall formation and reproductive development

## 🤖 Machine Learning Models

### 1. Decision Tree Classifier
- **Purpose**: Rule-based classification with interpretable results
- **Advantages**: Easy to understand, handles non-linear relationships
- **Use Case**: When you need to explain prediction reasoning

### 2. Random Forest
- **Purpose**: Ensemble method combining multiple decision trees
- **Advantages**: Reduces overfitting, handles missing values
- **Use Case**: When accuracy is prioritized over interpretability

### 3. K-Nearest Neighbors (KNN)
- **Purpose**: Classification based on similarity to neighboring samples
- **Advantages**: Simple, effective for small datasets
- **Use Case**: When local patterns in data are important

### 4. Apriori Algorithm
- **Purpose**: Association rule mining for pattern discovery
- **Advantages**: Finds relationships between soil parameters
- **Use Case**: Understanding which parameters co-occur in fertile soils

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Poetry (for dependency management)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/RedhaWassim/DataMining-project.git
cd DataMining-project
```

2. **Install Poetry** (if not already installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Install dependencies**
```bash
poetry install
```

4. **Activate virtual environment**
```bash
poetry shell
```

5. **Run the application**
```bash
streamlit run gui.py
```

## 🎮 Usage

### Starting the Dashboard
```bash
# Activate environment
poetry shell

# Launch Streamlit app
streamlit run gui.py
```

### Training Models
```python
from soil_fertility.components.data_ingestion import DataIngestion
from soil_fertility.components.model_trainer import ModelTrainer

# Load and process data
ingestion = DataIngestion()
train_path, test_path = ingestion.init_ingestion(path="your_data.csv")

# Train models
trainer = ModelTrainer()
results = trainer.init_training(train_data, test_data, "Fertility")
```

### Making Predictions
```python
from soil_fertility.pipeline.predict_pipeline import PredictPipeline, InputData

# Create input data
soil_sample = InputData(
    N=240, P=32, K=280, pH=6.5, EC=0.8, OC=1.2,
    S=15, Zn=2.5, Fe=45, Cu=1.8, Mn=25, B=0.8, OM=2.1
)

# Get prediction
pipeline = PredictPipeline()
fertility_level = pipeline.predict(
    soil_sample.get_data_as_df(), 
    "random_forest.pkl"
)
```

## ⚙️ How It Works

### 1. Data Ingestion Process
```
Raw Data (CSV/Excel) → Data Validation → General Processing → Train/Test Split
```

### 2. Model Training Workflow
```
Training Data → Feature Engineering → Model Training → Hyperparameter Tuning → Model Evaluation → Model Persistence
```

### 3. Prediction Pipeline
```
New Soil Sample → Data Preprocessing → Feature Transformation → Model Prediction → Fertility Classification
```

### 4. Web Interface Flow
```
User Input → Data Validation → Pipeline Processing → Results Display → Visualization
```

## 🧩 Project Components

### Core Components

#### `DataIngestion`
- Loads soil data from multiple formats
- Performs initial data validation and cleaning
- Splits data into training and testing sets
- Handles missing values and data quality issues

#### `ModelTrainer`
- Orchestrates training of multiple ML models
- Supports both default and GridSearch modes
- Evaluates model performance using various metrics
- Saves trained models for future use

#### `PredictPipeline`
- Loads trained models and preprocessors
- Processes new soil samples for prediction
- Returns fertility classification results
- Handles real-time prediction requests

#### `InputData`
- Validates input soil parameters
- Converts input to proper DataFrame format
- Ensures data consistency for predictions
- Provides type safety for soil measurements

### Model Implementations

Each model is implemented as a separate class with:
- Configurable hyperparameters
- Fit and predict methods
- Model-specific optimization
- Performance evaluation capabilities

### Data Transformation

The data transformation pipeline includes:
- **Feature Scaling**: Normalizing parameter ranges
- **Missing Value Handling**: Imputation strategies
- **Feature Engineering**: Creating derived features
- **Data Quality Checks**: Ensuring data integrity

## ⚙️ Configuration

### Model Parameters (`models_params.yaml`)
```yaml
# Example configuration
decision_tree:
  max_depth: [5, 10, 15]
  min_samples_split: [2, 5, 10]

random_forest:
  n_estimators: [50, 100, 200]
  max_depth: [10, 20, None]
```

### Path Configuration
The system uses configurable paths for:
- Raw data storage
- Processed data artifacts
- Trained model persistence
- Preprocessing objects

## 🔧 Development

### Code Quality Tools

**Formatting and Linting:**
```bash
# Format code
poetry run black soil_fertility/

# Lint code
poetry run ruff check soil_fertility/

# Type checking
poetry run mypy soil_fertility/

# Spell checking
poetry run codespell
```

**Testing:**
```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=soil_fertility
```

### Development Workflow

1. **Setup**: Install dependencies with Poetry
2. **Code**: Follow type hints and documentation standards
3. **Test**: Write tests for new functionality
4. **Quality**: Run formatting and linting tools
5. **Commit**: Use clear, descriptive commit messages

### Adding New Models

To add a new ML model:

1. Create model class in `soil_fertility/components/models/`
2. Implement required methods (`fit`, `predict`)
3. Add model to `ModelTrainer` class
4. Update configuration in `models_params.yaml`
5. Add tests for the new model

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Make your changes following the code style
4. Add tests for new functionality
5. Run quality checks (`make lint`, `make test`)
6. Commit your changes (`git commit -m 'Add new ML model'`)
7. Push to the branch (`git push origin feature/new-model`)
8. Open a Pull Request

## 📊 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate per class
- **Recall**: Sensitivity for each fertility level
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the poetry environment
poetry shell
```

**Data Format Issues:**
- Ensure CSV/Excel files have proper column headers
- Check for missing values in required parameters
- Verify numeric data types for soil measurements

**Model Training Fails:**
- Check data quality and completeness
- Verify sufficient samples for training
- Review parameter configurations in YAML file

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Redha Wassim**
- Email: bra.rwassim@gmail.com
- GitHub: [@RedhaWassim](https://github.com/RedhaWassim)

---

<div align="center">
<p><strong>Built with Python • Scikit-learn • Streamlit • Poetry</strong></p>
<p><em>Making soil fertility analysis accessible through data mining</em></p>
</div>
=======
The files as well as the source code of this project are found in the "master" branch

# AI-powered-information-retrieval-system
I developed an AI-powered information retrieval system designed to efficiently search and extract relevant data from large datasets. The system leverages natural language processing (NLP) and machine learning algorithms to understand user queries, retrieve precise information, and provide accurate responses.
>>>>>>> 7e8bcd5bba3ee5d7d125e81537f4a81a0a2e1ed0
=======
In this project, I processed and analyzed a dataset containing soil property data. Initially, I performed data preprocessing to clean and consolidate the information. Moving forward, I applied classification techniques to analyze soil fertility, categorizing different types of soil based on their characteristics. Additionally, I used clustering methods to group similar soil properties, allowing for the identification of patterns and trends within the dataset. This comprehensive approach provided valuable insights into the soil's characteristics, enabling informed conclusions and recommendations for agricultural applications.
>>>>>>> 49f58b505e5dd90858d9b4527abfeaf40bc0774c
