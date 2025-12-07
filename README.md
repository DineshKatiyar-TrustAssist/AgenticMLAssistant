# Agentic AI - ML Assistant

An intelligent multi-agent system that automates machine learning workflows using Google's Agent Development Kit (ADK). This application provides a Streamlit-based UI for automated data science tasks including exploratory data analysis (EDA), data cleaning, feature engineering, and model training.

## 🚀 Features

- **Multi-Agent Pipeline**: Automated workflow using specialized AI agents
  - **Planner Agent**: Breaks down tasks into structured plans
  - **Researcher Agent**: Researches best approaches and algorithms
  - **Coder Agent**: Generates clean, executable Python code
  - **Reviewer Agent**: Reviews and improves code quality
  - **Code Execution Agent**: Executes code and provides comprehensive summaries

- **Streamlit Web Interface**: User-friendly UI for:
  - CSV file upload
  - Google API key input
  - Target variable specification
  - Model selection (Supervised and Unsupervised)
  - Real-time results display

- **Supported ML Models**:
  - **Supervised**: RandomForest, LogisticRegression, XGBoost, SVM, KNN, NaiveBayes, AdaBoost, ExtraTrees, GradientBoosting, DecisionTree
  - **Unsupervised**: KMeans, DBSCAN, AgglomerativeClustering, MeanShift

- **Automated Workflow**:
  - Data loading and inspection
  - Exploratory Data Analysis (EDA)
  - Data cleaning and preprocessing
  - Feature engineering
  - Model training and evaluation
  - Comprehensive execution summaries

## 📋 Requirements

- Python 3.12+
- Google API Key (Gemini API)
- See `requirements.txt` for full dependency list

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AgenticMLAssistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a Google API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - You'll enter this in the Streamlit UI (no .env file needed)

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the UI

1. **Enter Google API Key**: 
   - Input your Google API key in the sidebar (password field)

2. **Upload CSV File**:
   - Click "Upload CSV file" and select your dataset

3. **Specify Target Variable**:
   - Enter the name of the target/prediction column (e.g., "Survived", "Class", "Price")

4. **Select ML Model**:
   - Choose from **Supervised ML Model** dropdown (for classification/regression)
   - OR choose from **Unsupervised ML Model** dropdown (for clustering)
   - Select only one model from either category

5. **Run Pipeline**:
   - Click "🚀 Run ML Model" button
   - Wait for the multi-agent pipeline to complete

6. **View Results**:
   - Execution results with metrics and summaries
   - Generated code (expandable)
   - Execution plan (expandable)

## 🏗️ Architecture

### Multi-Agent System Flow

```
User Task
    ↓
Planner Agent → Creates structured plan
    ↓
Researcher Agent → Researches approaches (uses {plan})
    ↓
Coder Agent → Generates Python code (uses {research})
    ↓
Reviewer Agent → Reviews and improves code (uses {code})
    ↓
Code Execution Agent → Executes code and creates summary (uses {reviewed_code})
    ↓
Results Display
```

### Key Components

- **`ml_ai_agents.py`**: Core multi-agent system implementation
  - Agent definitions and configurations
  - Code extraction and execution functions
  - Pipeline orchestration using SequentialAgent

- **`app.py`**: Streamlit web interface
  - File upload and configuration
  - Results parsing and display
  - User interaction handling

## 📦 Dependencies

```
python-dotenv>=1.0.0,<2.0.0
google-adk>=0.1.0
scikit-learn>=1.3.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
seaborn>=0.12.0,<1.0.0
streamlit>=1.28.0
```

## 🔧 Configuration

### API Key Management

The application supports API key input via:
- **UI Input** (Recommended): Enter directly in the Streamlit sidebar
- **Environment Variable** (Optional): Set `GOOGLE_API_KEY` in `.env` file

**Note**: API key entered in UI takes precedence over `.env` file.

### Model Selection

- **Supervised Models**: Use for classification or regression tasks
  - Requires target variable
  - Provides accuracy, ROC-AUC, classification reports

- **Unsupervised Models**: Use for clustering tasks
  - No target variable needed (though UI still requires it for now)
  - Provides cluster analysis and visualizations

## 📊 Output Format

The execution summary includes:

1. **Data Loading**:
   - Dataset shape and basic statistics
   - First 5 rows preview
   - Data types information

2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics
   - Missing values analysis

3. **Data Cleaning & Preprocessing**:
   - Target variable identification
   - Feature engineering steps
   - Imputation methods

4. **Model Training**:
   - Model type and configuration
   - Train/test split information

5. **Results**:
   - Model accuracy
   - ROC-AUC score (for classification)
   - Classification report
   - Confusion matrix
   - Feature importances (top 5)

## 🐛 Troubleshooting

### Common Issues

1. **"Context variable not found: reviewed_code"**
   - This has been resolved in the latest version
   - The code execution agent now handles missing variables gracefully

2. **"Unclosed connector" warnings**
   - These are informational and don't affect functionality
   - Suppressed in the current version

3. **API Key Errors**
   - Ensure your Google API key is valid
   - Check that you have access to Gemini API

4. **Model Training Errors**
   - Verify your target variable name matches a column in the CSV
   - Ensure the dataset has sufficient data
   - Check that the target variable has appropriate values for the selected model type

## 📝 Example Usage

1. **Classification Task**:
   - Upload: `titanic.csv`
   - Target Variable: `Survived`
   - Model: `RandomForestClassifier`
   - Result: Classification model with accuracy metrics

2. **Regression Task**:
   - Upload: `housing.csv`
   - Target Variable: `Price`
   - Model: `LinearRegression`
   - Result: Regression model with R² and MSE metrics

3. **Clustering Task**:
   - Upload: `customer_data.csv`
   - Target Variable: `CustomerID` (or any column)
   - Model: `KMeans`
   - Result: Cluster analysis with silhouette scores

## 🔒 Security Notes

- API keys are entered as password fields (hidden input)
- Temporary CSV files are automatically cleaned up after execution
- No data is stored permanently on the server

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Dinesh Katiyar

## 🙏 Acknowledgments

- Google ADK (Agent Development Kit) for multi-agent orchestration
- Streamlit for the web interface framework
- scikit-learn for machine learning models

## 📚 Additional Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

**Note**: This is an automated ML assistant. Always review the generated code and results before using in production environments.

