# Agentic AI - ML Assistant

An intelligent multi-agent system that automates machine learning workflows using Google's Agent Development Kit (ADK). This application provides a modern web interface built with React/Next.js frontend and FastAPI backend for automated data science tasks including exploratory data analysis (EDA), data cleaning, feature engineering, and model training.

## 🚀 Features

- **Multi-Agent Pipeline**: Automated workflow using specialized AI agents
  - **Planner Agent**: Breaks down tasks into structured plans
  - **Researcher Agent**: Researches best approaches and algorithms
  - **Coder Agent**: Generates clean, executable Python code
  - **Reviewer Agent**: Reviews and improves code quality
  - **Code Execution Agent**: Executes code and provides comprehensive summaries

- **Modern Web Interface**: React/Next.js frontend with:
  - Resizable and collapsible configuration sidebar
  - CSV file upload
  - Google API key input (password-protected)
  - Target variable specification
  - Model selection (Supervised and Unsupervised dropdowns)
  - Real-time loading indicators with spinning wheel
  - Structured results display with metrics, code blocks, and expandable sections
  - Color-coded section headers (light green for results, medium green for code/plan)

- **RESTful API**: FastAPI backend providing:
  - File upload handling (multipart/form-data)
  - Pipeline execution with async support
  - JSON responses
  - Health check endpoint
  - Configurable CORS support

- **Supported ML Models**:
  - **Supervised**: RandomForest, LogisticRegression, XGBoost, SVM, KNN, NaiveBayes, AdaBoost, ExtraTrees, GradientBoosting, DecisionTree
  - **Unsupervised**: KMeans, DBSCAN, AgglomerativeClustering, MeanShift

- **Automated Workflow**:
  - Data loading and inspection
  - Exploratory Data Analysis (EDA)
  - Data cleaning and preprocessing
  - Feature engineering
  - Model training and evaluation
  - Comprehensive execution summaries with structured output

- **Docker Support**:
  - Separate container deployment (backend + frontend)
  - Combined container deployment (single container)
  - Production-ready configurations
  - Health checks and graceful shutdown

## 📋 Requirements

- Python 3.12+
- Node.js 18+
- Google API Key (Gemini API)

## 🛠️ Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server**:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   Backend will run on `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the Next.js development server**:
   ```bash
   npm run dev
   ``` 

   Frontend will run on `http://localhost:3000`

## 🎯 Usage

### Option 1: Docker Deployment (Recommended)

#### Combined Container (Single Container)

Deploy both frontend and backend in a single container:

```bash
# Build the combined container
docker build -t agentic-ml-assistant .

# Run the container
docker run -d -p 8000:8000 -p 3000:3000 --name agentic-ml agentic-ml-assistant

# Or using docker-compose
docker-compose -f docker-compose.combined.yml up -d
```

#### Separate Containers

Deploy frontend and backend as separate containers:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access the application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

See [DOCKER.md](./DOCKER.md) for detailed Docker deployment instructions.

### Option 2: Local Development

1. **Start both servers** (backend and frontend)
2. **Open your browser** to `http://localhost:3000`
3. **Enter Google API Key**: Input your Google API key in the sidebar
4. **Upload CSV File**: Click "Upload Data/CSV file" and select your dataset
5. **Specify Target Variable**: Enter the name of the target/prediction column
6. **Select ML Model**: Choose from Supervised or Unsupervised ML Model dropdowns
7. **Run Pipeline**: Click "🚀 Run ML Model" button
8. **View Results**: 
   - Execution results with metrics and summaries
   - Generated code (expandable)
   - Execution plan (expandable)

## 🏗️ Architecture

### Project Structure

```
AgenticMLAssistant/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── ml_ai_agents.py      # Multi-agent system implementation
│   ├── requirements.txt     # Backend dependencies
│   └── Dockerfile           # Backend Docker configuration
├── frontend/
│   ├── pages/
│   │   └── index.tsx        # Next.js main page
│   ├── package.json         # Frontend dependencies
│   ├── tsconfig.json        # TypeScript configuration
│   ├── next.config.js       # Next.js configuration
│   └── Dockerfile           # Frontend Docker configuration
├── Dockerfile               # Combined container Dockerfile
├── docker-compose.yml       # Docker Compose for separate containers
├── docker-compose.combined.yml  # Docker Compose for combined container
├── .dockerignore            # Docker ignore patterns
├── DOCKER.md                # Docker deployment guide
└── README.md                # This file
```

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
Code Execution Agent → Executes code and creates summary
    ↓
Results Display (via FastAPI → React Frontend)
```

### Key Components

- **`backend/ml_ai_agents.py`**: Core multi-agent system implementation
  - Agent definitions and configurations
  - Code extraction function (`extract_python_code`)
  - Code execution function (`execute_code`) with error handling
  - Pipeline orchestration using SequentialAgent
  - State management and event processing

- **`backend/main.py`**: FastAPI REST API
  - File upload handling
  - Pipeline execution endpoint
  - CORS configuration

- **`frontend/pages/index.tsx`**: React/Next.js web interface
  - Resizable/collapsible sidebar with drag handle
  - File upload and configuration
  - Results parsing and structured display
  - Loading indicators with CSS animations
  - Error handling and user feedback
  - Environment variable support for API URL

## 📦 Dependencies

### Backend (`backend/requirements.txt`)

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
python-dotenv>=1.0.0,<2.0.0
google-adk>=0.1.0
scikit-learn>=1.3.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0,<4.0.0
seaborn>=0.12.0,<1.0.0
```

### Frontend (`frontend/package.json`)

```
next>=14.0.0
react>=18.2.0
react-dom>=18.2.0
axios>=1.6.0
```

## 🔧 Configuration

### API Key Management

The application supports API key input via:
- **UI Input** (Recommended): Enter directly in the web interface
- **Environment Variable** (Optional): Set `GOOGLE_API_KEY` in `.env` file

**Note**: API key entered in UI takes precedence over `.env` file.

### Model Selection

- **Supervised Models**: Use for classification or regression tasks
  - Requires target variable
  - Provides accuracy, ROC-AUC, classification reports

- **Unsupervised Models**: Use for clustering tasks
  - Requires target variable (for UI consistency)
  - Provides cluster analysis and visualizations

## 📡 API Endpoints

### `POST /api/run-pipeline`

Run the multi-agent ML pipeline.

**Request** (multipart/form-data):
- `file`: CSV file
- `api_key`: Google API key
- `target_variable`: Target column name
- `model_name`: Selected ML model name

**Response**:
```json
{
  "plan": "...",
  "research": "...",
  "code": "...",
  "reviewed_code": "...",
  "execution_result": "...",
  "final_response": "..."
}
```

### `GET /api/health`

Health check endpoint.

**Response**:
```json
{
  "status": "ok"
}
```

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

1. **CORS Errors**
   - Ensure backend is running on port 8000
   - Check CORS configuration in `backend/main.py`
   - For Docker deployments, set `CORS_ORIGINS` environment variable

2. **Docker Build Failures**
   - Ensure Docker and Docker Compose are installed
   - Check that all required files are present
   - Review Docker logs: `docker-compose logs`

3. **Frontend Can't Connect to Backend**
   - Verify `NEXT_PUBLIC_API_URL` is set correctly
   - For Docker, use service names (e.g., `http://backend:8000`)
   - Check network connectivity between containers

4. **API Key Errors**
   - Ensure your Google API key is valid
   - Check that you have access to Gemini API
   - Verify the API key is entered correctly in the UI

5. **Model Training Errors**
   - Verify your target variable name matches a column in the CSV
   - Ensure the dataset has sufficient data
   - Check that the target variable has appropriate values for the selected model type
   - Review execution logs for detailed error messages

6. **Docker Container Issues**
   - Check container status: `docker ps -a`
   - View container logs: `docker logs <container-name>`
   - Restart containers: `docker-compose restart`

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
- CORS is configurable via environment variables
- Docker containers run as non-root users
- Health checks ensure service availability

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Dinesh Katiyar

## 🙏 Acknowledgments

- Google ADK (Agent Development Kit) for multi-agent orchestration
- FastAPI for the REST API framework
- Next.js and React for the frontend framework
- scikit-learn for machine learning models
- Docker for containerization support

## 📚 Additional Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---
**Note**: This is an automated ML assistant. Always review the generated code and results before using in production environments.
