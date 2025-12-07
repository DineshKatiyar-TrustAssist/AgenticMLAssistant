"""
Streamlit UI for Agentic ML Assistant
"""
import streamlit as st
import asyncio
import os
import tempfile
import re
from ml_ai_agents import run_pipeline


def parse_and_display_results(result_text):
    """Parse execution results and display in structured format."""
    if not result_text or result_text == "N/A":
        return
    
    sections = re.split(r'^--- (.+?) ---$', result_text, flags=re.MULTILINE)
    if sections and not sections[0].strip() and len(sections) > 1:
        sections = sections[1:]
    
    if len(sections) <= 1:
        st.text_area("Execution Output", result_text, height=400)
        return
    
    icons = {"Data Loading": "📊", "EDA": "🔍", "Exploratory": "🔍", 
             "Data Cleaning": "🧹", "Preprocessing": "🧹", 
             "Model Training": "🤖", "Evaluation": "🤖", "Evaluation Results": "📈"}
    
    current_section = None
    for i, part in enumerate(sections):
        if i % 2 == 1:
            current_section = part.strip()
        elif i % 2 == 0 and current_section and part.strip():
            content = part.strip()
            icon = next((icons[k] for k in icons if k in current_section), "")
            st.subheader(f"{icon} {current_section}" if icon else current_section)
            
            if "Data Loading" in current_section:
                if m := re.search(r'Dataset Shape: \((\d+), (\d+)\)', content):
                    st.metric("Dataset Shape", f"{m.group(1)} rows × {m.group(2)} columns")
                for label, key in [("First 5 Rows:", "First 5 Rows:"), ("Data Types:", "Data Types:")]:
                    if key in content:
                        st.text(label)
                        st.code(content.split(key)[1].split("Data Types:" if label == "First 5 Rows:" else "---")[0].strip())
            
            elif "EDA" in current_section or "Exploratory" in current_section:
                for label, key in [("Summary Statistics:", "Summary Statistics"), ("Missing Values per Column:", "Missing Values")]:
                    if key in content:
                        st.text(label)
                        st.code(content.split(key)[1].split("Missing Values" if key == "Summary Statistics" else "---")[0].strip())
            
            elif "Evaluation Results" in current_section:
                col1, col2 = st.columns(2)
                if m := re.search(r'Accuracy: ([\d.]+)', content):
                    col1.metric("Accuracy", f"{float(m.group(1)):.4f}")
                if m := re.search(r'ROC-AUC Score.*?: ([\d.]+)', content):
                    col2.metric("ROC-AUC Score", f"{float(m.group(1)):.4f}")
                for label, key, end in [("Classification Report:", "Classification Report:", "Confusion Matrix"), 
                                        ("Confusion Matrix:", "Confusion Matrix:", "")]:
                    if key in content:
                        st.text(label)
                        st.code(content.split(key)[1].split(end)[0].strip() if end else content.split(key)[1].strip())
            
            else:
                st.text(content)

# Set environment variable to prevent direct execution
os.environ["STREAMLIT_RUNNING"] = "1"

st.set_page_config(page_title="Agentic ML Assistant", layout="wide")

st.title("Agentic AI - ML Assistant")
st.markdown("Upload a CSV file and select an ML model for automated training and evaluation")

# Supervised ML Model options
SUPERVISED_MODELS = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "LogisticRegression",
    "LinearRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "SVC",
    "SVR",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "NaiveBayes",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
]

# Unsupervised ML Model options
UNSUPERVISED_MODELS = [
    "KMeans",
    "DBSCAN",
    "AgglomerativeClustering",
    "MeanShift",
]

# Sidebar for inputs
with st.sidebar:
    st.header("📋 Configuration")
    
    # Google API Key input
    api_key = st.text_input("GOOGLE_API_KEY", type="password", help="Enter your Google API key")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Target variable input
    target_variable = st.text_input("Target Column/Prediction Variable")
    
    # Model selection dropdowns
    st.subheader("Model Selection")
    supervised_model = st.selectbox("Supervised ML Model", ["None"] + SUPERVISED_MODELS, index=0)
    unsupervised_model = st.selectbox("Unsupervised ML Model", ["None"] + UNSUPERVISED_MODELS, index=0)
    
    # Run button
    run_button = st.button("🚀 Run ML Model", type="primary")

# Main content area
# Determine selected model
model_name = None
if supervised_model and supervised_model != "None":
    model_name = supervised_model
elif unsupervised_model and unsupervised_model != "None":
    model_name = unsupervised_model

if uploaded_file and target_variable and model_name and api_key and run_button:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        csv_path = tmp_file.name
    
    # Set API key from UI (overrides .env file)
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    
    # Create task for pipeline
    task = f"""
    Load the CSV file from {csv_path}, perform Exploratory Data Analysis (EDA),
    clean the data, preprocess features, use '{target_variable}' as the target variable,
    train a {model_name}, evaluate the model using accuracy and other metrics, and output results.
    """
    
    # Show progress
    with st.spinner("🔄 Running multi-agent pipeline..."):
        try:
            result = asyncio.run(run_pipeline(task))
            
            # Display results
            st.success("✅ Pipeline completed successfully!")
            
            # Get execution result
            exec_result = result.get("execution_result", "N/A")
            
            # Extract actual output from execution result
            if exec_result != "N/A":
                output_text = ""
                
                # Handle different result formats
                if isinstance(exec_result, dict):
                    output_text = exec_result.get("message", "")
                else:
                    output_text = str(exec_result)
                    # Extract stdout output if embedded in agent response
                    # Look for the actual output starting with "---" sections
                    if "---" in output_text:
                        # Extract everything from first "---" section
                        match = re.search(r'(--- .+? ---.*)', output_text, re.DOTALL)
                        if match:
                            output_text = match.group(1)
                    # Or look for common output patterns
                    elif "Dataset Shape:" in output_text or "Accuracy:" in output_text:
                        # This is likely the actual output
                        pass
                    else:
                        # Try to extract from dict-like string
                        match = re.search(r'["\']message["\']:\s*["\'](.*?)["\']', output_text, re.DOTALL)
                        if match:
                            output_text = match.group(1).replace('\\n', '\n')
                
                # Strip asterisks from execution summary
                output_text = re.sub(r'\*{1,3}', '', output_text)
                
                # Parse and display structured results
                if output_text:
                    parse_and_display_results(output_text)
            
            # Display reviewed code
            with st.expander("📝 Generated Code"):
                if result.get("reviewed_code") and result["reviewed_code"] != "N/A":
                    st.code(result["reviewed_code"], language="python")
            
            # Display plan
            with st.expander("📋 Execution Plan"):
                if result.get("plan") and result["plan"] != "N/A":
                    st.markdown(result["plan"])
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(csv_path):
                os.unlink(csv_path)
                
elif run_button:
    missing = []
    if not api_key:
        missing.append("Google API Key")
    if not uploaded_file:
        missing.append("CSV file")
    if not target_variable:
        missing.append("target variable")
    if not model_name:
        missing.append("a model (select from either Supervised or Unsupervised)")
    st.warning(f"⚠️ Please provide: {', '.join(missing)}")

else:
    st.info("👈 Enter your Google API key, upload a CSV file, enter a target variable, and select a model from either Supervised or Unsupervised ML models in the sidebar to get started")

