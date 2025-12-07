
"""
Agentic ML Assistant

This module implements a multi-agent system that automates repetitive data science tasks
including exploratory data analysis (EDA), data cleaning, feature engineering, and
initial model development. The system uses Google ADK (Agent Development Kit) to create
a sequence of specialized agents:

1. Planner Agent: Breaks down tasks into structured plans
2. Researcher Agent: Researches approaches and algorithms
3. Coder Agent: Generates Python code to implement solutions
4. Reviewer Agent: Reviews and improves code quality

The system uses Google ADK's SequentialAgent to orchestrate agents in a deterministic
pipeline, ensuring reliable execution order.

Configuration:
    - Create a .env file in the project root with: GOOGLE_API_KEY=your_api_key_here
    - Install dependencies: pip install -r requirements.txt

Author: Dinesh Katiyar
License: MIT
"""

# # Auto-ML Multi-Agent Data Scientist
# 
# **Problem statement:** Data scientists and analysts spend a lot of time on repeated tasks — 
# exploratory data analysis (EDA), cleaning, feature engineering, and producing an initial model. 
# This script demonstrates a multi-agent system that automates those repetitive steps: an agent 
# plans the pipeline, another researches approaches, a coder agent writes the code, a reviewer 
# agent checks/fixes it, and an executor runs the code.
# 
# **Goals & deliverables:**
# - Demonstrate a multi-agent sequence (Planner → Researcher → Coder → Reviewer → Executor).
# - Show ADK usage for planning and code generation (Gemini API).
# - Execute generated code safely and display results (EDA summary and a basic model metric).
# 
# **How to use:** Create a `.env` file with `GOOGLE_API_KEY=your_key` before running.

# In[1]:


# ============================================================================
# SECTION 1: Setup and Configuration
# ============================================================================
# This section handles:
# - Loading API keys from .env file
# - Configuring Google ADK and Gemini API
# - Setting up logging
# ============================================================================

from dotenv import load_dotenv
import os
import logging
import asyncio
from pprint import pprint
import warnings
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import ast

# Load environment variables from .env file
# This will look for .env in the current directory and parent directories
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auto_ml_agents")

# Suppress aiohttp unclosed session/connector warnings and errors
class AiohttpFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(x in msg for x in ["Unclosed connector", "Unclosed client session", "Unclosed client"])
        
for log_name in ["aiohttp", "aiohttp.client", "aiohttp.connector"]:
    log = logging.getLogger(log_name)
    log.setLevel(logging.CRITICAL)
    log.addFilter(AiohttpFilter())

# Attempt to load API key from .env file (optional - can be provided via UI)
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    logger.info("✅ Loaded GOOGLE_API_KEY from .env file.")
else:
    logger.info("ℹ️ No GOOGLE_API_KEY found in .env file. It can be provided via UI.")


# In[2]:


# ============================================================================
# SECTION 2: Import Google ADK Components
# ============================================================================
# Import the necessary components from Google ADK for building multi-agent systems.
# ============================================================================

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.events import Event
from google.adk.tools import FunctionTool
from google.genai import types

logger.info("✅ Google ADK components imported successfully.")


# In[3]:


# ============================================================================
# SECTION 3: Agent Definitions
# ============================================================================
# Define specialized agents using Google ADK's Agent class.
# Each agent has a specific role and uses output_key to pass state to the next agent.
# ============================================================================

# Planner Agent: Breaks down tasks into structured plans
planner_agent = Agent(
    name="PlannerAgent",
    model="gemini-3-pro-preview",
    instruction="""You are a specialized planning agent for data science projects. 
    Your job is to break down the user's task into clear, structured steps.
    
    When given a task, create a detailed plan that includes:
    1. Data loading and exploration steps
    2. Data cleaning and preprocessing requirements
    3. Feature engineering approaches
    4. Model selection and training steps
    5. Evaluation and output requirements
    
    Present your plan in a clear, numbered format that can guide the next agents.""",
    output_key="plan",  # Output stored in session state with this key
)

logger.info("✅ Planner agent created.")


# Researcher Agent: Researches approaches and algorithms
researcher_agent = Agent(
    name="ResearchAgent",
    model="gemini-3-pro-preview",
    instruction="""You are a specialized research agent for data science. 
    Your job is to research and recommend the best approaches, algorithms, and techniques.
    
    Given this plan: {plan}
    
    Research and recommend:
    1. Appropriate algorithms for the task (classification, regression, etc.)
    2. Best practices for data preprocessing
    3. Feature engineering techniques
    4. Model evaluation metrics
    5. Libraries and tools to use
    
    Provide specific, actionable recommendations with brief explanations.""",
    output_key="research",  # Output stored in session state with this key
)

logger.info("✅ Researcher agent created.")


# Coder Agent: Generates Python code
coder_agent = Agent(
    name="CoderAgent",
    model="gemini-3-pro-preview",
    instruction="""You are a specialized coding agent for data science. 
    Your job is to write clean, efficient, and well-commented Python code.
    
    Based on this research: {research}
    
    Write complete, runnable Python code that:
    1. Implements the data loading and EDA steps
    2. Performs data cleaning and preprocessing
    3. Implements feature engineering
    4. Trains the recommended model
    5. Evaluates the model and displays results
    
    CRITICAL REQUIREMENTS - AVOID COMMON DATA ERRORS:
    
    Data Type Handling:
    - ALWAYS check data types (df.dtypes) before converting columns
    - NEVER convert string/text columns to float or int directly
    - Use pd.to_numeric() with errors='coerce' for numeric conversion, then handle NaN values
    - For categorical columns, use LabelEncoder or OneHotEncoder, NOT direct type conversion
    - Check for non-numeric values in numeric columns before conversion
    
    Model Training & Validation:
    - ALWAYS verify model is properly initialized before use (model is not None)
    - Check that model.fit() completed successfully before calling predict()
    - Use try-except blocks around model training and evaluation
    - Verify X_train and y_train are not empty before training
    - Ensure all features are numeric before passing to model (no string columns)
    
    Data Preprocessing:
    - Handle missing values BEFORE model training (dropna, fillna, or imputation)
    - Encode categorical variables properly (LabelEncoder, OneHotEncoder, or pd.get_dummies)
    - Remove or handle non-numeric columns before training
    - Verify data shapes match (X_train.shape[0] == y_train.shape[0])
    - ALWAYS reset index after dropna/filtering: df = df.reset_index(drop=True) to prevent IndexError
    
    Error Prevention:
    - Add validation checks: if model is None, raise clear error
    - Check data types before operations: if df[col].dtype == 'object', handle appropriately
    - Use pd.api.types.is_numeric_dtype() to check if column is numeric
    - Validate model objects before calling methods (hasattr(model, 'predict'))
    - Reset index after dropna/filtering to prevent IndexError: df.reset_index(drop=True)
    
    General Requirements:
    - Use pandas, numpy, scikit-learn, matplotlib/seaborn
    - Include proper error handling with try-except blocks
    - Add clear comments explaining each step
    - Make the code self-contained and executable
    - Output only the Python code, wrapped in ```python code blocks if needed""",
    output_key="code",  # Output stored in session state with this key
)

logger.info("✅ Coder agent created.")


# Reviewer Agent: Reviews and improves code
reviewer_agent = Agent(
    name="ReviewerAgent",
    model="gemini-3-pro-preview",
    instruction="""You are a specialized code review agent. 
    Your job is to review code for correctness, quality, and improvements.
    
    Review this code: {code}
    
    CRITICAL CHECKS - PREVENT COMMON DATA SCIENCE ERRORS:
    
    1. Data Type Conversion Errors:
       - Verify NO direct conversion of string/text columns to float/int (e.g., df['Name'].astype(float))
       - Check that pd.to_numeric() is used with errors='coerce' for safe conversion
       - Ensure categorical columns are encoded (LabelEncoder/OneHotEncoder) before model training
       - Verify numeric columns don't contain non-numeric strings before conversion
    
    2. Model Validation Errors:
       - Ensure model is initialized and not None before use
       - Verify model.fit() is called and completes successfully
       - Check that model object exists before calling model.predict()
       - Add validation: if model is None, raise ValueError with clear message
       - Verify estimator parameter in cross_val_score is a valid model object, not None
    
    3. Data Preprocessing Issues:
       - Check that missing values are handled before model training
       - Verify all features are numeric before passing to model (no object/string dtypes)
       - Ensure X_train and y_train have matching row counts
       - Check that categorical variables are properly encoded
       - Verify index is reset after dropna/filtering: df.reset_index(drop=True) to prevent IndexError
    
    4. General Code Quality:
       - Syntax errors and logical issues
       - Missing imports or dependencies
       - Best practices and code quality
       - Potential bugs or edge cases
       - Code efficiency and optimization opportunities
       - Proper error handling with try-except blocks
    
    REQUIRED FIXES:
    - If you find data type conversion issues, fix them by:
      * Using pd.to_numeric(..., errors='coerce') for numeric conversion
      * Encoding categorical variables properly before training
      * Checking dtypes before conversion operations
    
    - If you find IndexError issues, fix them by:
      * Adding df.reset_index(drop=True) after dropna() or filtering operations
      * Using iloc with valid range: iloc[:len(df)] instead of hardcoded indices
    
    - If you find model validation issues, fix them by:
      * Adding checks: if model is None: raise ValueError("Model not trained")
      * Verifying model.fit() completed successfully
      * Adding try-except around model operations
    
    - Always ensure:
      * All features are numeric before model training
      * Missing values are handled appropriately
      * Data shapes are validated
    
    Output:
    - Provide ONLY the corrected Python code
    - Wrap code in ```python ... ``` blocks
    - Do NOT include explanatory text - only the code
    - Ensure the code is ready to execute without errors""",
    output_key="reviewed_code",  # Final reviewed code stored here
)

logger.info("✅ Reviewer agent created.")


# Extract Python Code Function: Extracts code from text
def extract_python_code(text: str) -> str:
    """
    Extract Python code from text, skipping explanatory prose.
    
    Args:
        text (str): Text that may contain Python code
        
    Returns:
        str: Extracted Python code, or empty string if none found
    """
    if not text or text == "N/A":
        return ""
    
    # 1. Try markdown code blocks first (most reliable)
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        code = max(code_blocks, key=len).strip()
        if any(k in code for k in ['import', 'def', 'class', '=', 'print']):
            return code
    
    # 2. Find first line that starts with Python keywords (skip prose)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip explanatory text patterns
        if re.match(r'^(Okay|Here|This|The|I have|Let me|Note:|Summary:|Issues?|Here\'s)', stripped, re.I):
            continue
        # Found code start
        if stripped.startswith(('import ', 'from ', 'def ', 'class ', '# ', 'print', 'if ', 'for ', 'try:', 'with ')):
            code = '\n'.join(lines[i:]).strip()
            # Validate it's code (has Python syntax)
            if any(k in code for k in ['import', 'def', 'class', '=', 'print', '(', ')']):
                return code
    
    # 3. No code found
    return ""


# Code Execution Function: Executes Python code safely
def execute_code(code: str) -> dict:
    """
    Execute Python code and return results.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        dict: Execution result with status and output/error message
    """
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # Clean code (remove any remaining markdown markers)
    code = code.replace("```python", "").replace("```", "").strip()
    
    if not code:
        return {"status": "error", "message": "No code provided"}
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Syntax check
        compile(code, '<string>', 'exec')
        
        # Execute code
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals())
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        if errors:
            return {"status": "error", "message": f"Execution error: {errors}"}
        
        return {"status": "success", "message": output or "Code executed successfully"}
    
    except SyntaxError as e:
        return {"status": "error", "message": f"Syntax error at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"status": "error", "message": f"Runtime error: {type(e).__name__}: {str(e)}"}


# Code Execution Agent: Executes generated code
code_execution_agent = Agent(
    name="CodeExecutionAgent",
    model="gemini-3-pro-preview",
    instruction="""You are a code execution agent. Your job is to execute Python code and provide a comprehensive execution summary.
    
    Given this code: {reviewed_code}
    
    IMPORTANT: Follow these steps:
    1. First, call the extract_python_code tool with the reviewed_code to extract clean Python code
    2. Then, call the execute_code tool with the extracted code to execute it
    3. Analyze the execution output and create a detailed execution summary
    4. Add the execution summary to execution_result in the following format:
    
    EXECUTION SUMMARY FORMAT:
    After successful execution, provide a structured summary in the following format:
    
    ### **Execution Summary**
    
    1. **Data Loading**:
       * File path and loading status
       * Dataset shape (rows, columns)
    
    2. **Preprocessing**:
       * Target variable identification
       * Data cleaning steps (duplicates, high-cardinality features dropped)
       * Feature engineering steps
       * Imputation methods used
       * Final feature set (samples, features)
    
    3. **Model Training**:
       * Model type/name
       * Train/test split ratio
       * Any important notes about performance or data issues
    
    ### **Results**
    
    * Model Accuracy
    * Weighted F1-Score (if available)
    * Classification Report (formatted as code block)
    * Top 5 Feature Importances (if available, formatted as numbered list)
    
    If extraction fails or no code is found, report that. If execution fails, provide clear error messages.""",
    output_key="execution_result",
    tools=[FunctionTool(extract_python_code), FunctionTool(execute_code)],
)

logger.info("✅ Code execution agent created.")


# In[4]:


# ============================================================================
# SECTION 4: Multi-Agent System Orchestration
# ============================================================================
# Create a SequentialAgent that runs all agents in order.
# Each agent's output is automatically passed to the next via output_key references.
# ============================================================================

# Create the sequential multi-agent system
# Agents run in order: Planner -> Researcher -> Coder -> Reviewer -> CodeExecution
multi_agent_system = SequentialAgent(
    name="DataSciencePipeline",
    sub_agents=[planner_agent, researcher_agent, coder_agent, reviewer_agent, code_execution_agent],
)

logger.info("✅ Multi-Agent System (SequentialAgent) created successfully.")


# In[5]:


# ============================================================================
# SECTION 5: Task Definition and Execution
# ============================================================================
# Define the user task and execute the multi-agent system pipeline.
# ============================================================================

# Define the task for the multi-agent system
# user_task = """
# Load the Titanic dataset, perform Exploratory Data Analysis (EDA),
# clean the data, preprocess features, train a RandomForestClassifier,
# evaluate the model using accuracy and other metrics, and output results.
# """

# ============================================================================
# SECTION 6: Run the Multi-Agent System
# ============================================================================
# Execute the complete pipeline using InMemoryRunner.
# The runner handles async execution and state management.
# ============================================================================

def extract_text_from_content(content):
    """Extract text from content, filtering out function_call and thought_signature parts."""
    if not content:
        return None
    if hasattr(content, 'parts') and content.parts:
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                return part.text
            elif isinstance(part, dict) and 'text' in part:
                return part['text']
    elif isinstance(content, dict):
        if 'parts' in content:
            for part in content['parts']:
                if isinstance(part, dict) and 'text' in part:
                    return part['text']
    return None


async def run_pipeline(task: str):
    """
    Execute the multi-agent pipeline on a given task.
    
    Args:
        task (str): The data science task to automate
    
    Returns:
        dict: Results from all agents including plan, research, code, and reviewed_code
    """
    runner = InMemoryRunner(agent=multi_agent_system)
    
    print("\n" + "="*70)
    print("🚀 Starting Multi-Agent Data Science Pipeline")
    print("="*70)
    print(f"\n📋 Task: {task.strip()}\n")
    

    # Run the pipeline and get the response
    # Use run_debug to properly handle function calls and get full response
    # Suppress warnings during execution
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*App name mismatch.*')
            warnings.filterwarnings('ignore', message='.*app name.*')
            warnings.filterwarnings('ignore', message='.*non-text parts.*')
            # run_debug properly handles function calls and returns full response
            response = await runner.run_debug(task)
    except Exception as e:
        # If async fails, try to get more details about the error
        error_msg = str(e)
        raise RuntimeError(f"Failed to execute workflow: {error_msg}")
    finally:
        # Clean up runner resources to close aiohttp sessions
        try:
            if hasattr(runner, 'close'):
                await runner.close()
            elif hasattr(runner, '__aexit__'):
                await runner.__aexit__(None, None, None)
        except Exception:
            pass  # Ignore cleanup errors


    # Response is a list of Event objects according to ADK documentation
    # https://google.github.io/adk-docs/events/#identifying-event-origin-and-type
    if not isinstance(response, (list, tuple)):
        logger.warning(f"Expected list of Events, got {type(response)}")
        response = [response] if response else []
    
    print(f"Received {len(response)} events from pipeline\n")
    
    # Initialize state dictionary to accumulate state_delta from events
    state = {}
    
    # Dictionary to store agent outputs (keyed by agent name)
    agent_outputs = {
        "PlannerAgent": [],
        "ResearchAgent": [],
        "CoderAgent": [],
        "ReviewerAgent": [],
        "CodeExecutionAgent": []
    }
    
    # Process each event to extract state and agent outputs
    for event in response:
        if not isinstance(event, Event):
            logger.debug(f"Skipping non-Event object: {type(event)}")
            continue
        
        # Extract state_delta from event.actions and merge into state
        if hasattr(event, 'actions') and event.actions:
            if hasattr(event.actions, 'state_delta') and event.actions.state_delta:
                state_delta = event.actions.state_delta
                if isinstance(state_delta, dict):
                    state.update(state_delta)
                elif hasattr(state_delta, '__dict__'):
                    state.update(state_delta.__dict__)
        
        # Extract agent outputs based on event author
        if hasattr(event, 'author') and event.author:
            author = event.author
            content_text = extract_text_from_content(event.content) if hasattr(event, 'content') else None
            
            # Store agent output if we have content
            if content_text and author in agent_outputs:
                # Check if this is a final response (non-partial)
                is_final = True
                if hasattr(event, 'partial'):
                    is_final = not event.partial
                elif hasattr(event, 'is_final_response'):
                    try:
                        is_final = event.is_final_response()
                    except:
                        is_final = True
                
                # For final responses, replace previous partial responses
                if is_final:
                    agent_outputs[author] = [content_text]
                else:
                    # For partial responses, append
                    agent_outputs[author].append(content_text)
    
    # Also try to access state through the runner's session
    if not state:
        try:
            if hasattr(runner, 'session') and hasattr(runner.session, 'state'):
                runner_state = runner.session.state
                if isinstance(runner_state, dict):
                    state = runner_state
                elif hasattr(runner_state, '__dict__'):
                    state = runner_state.__dict__
        except Exception as e:
            logger.debug(f"Could not access runner session state: {e}")
    
    # Extract results from state and agent outputs
    # Helper function to safely get values
    def get_value(key, default="N/A"):
        """Extract value from state or agent outputs"""
        # First try state
        if isinstance(state, dict) and key in state:
            value = state[key]
            if value and value != "N/A":
                return value
        
        # Then try agent outputs based on key mapping
        key_to_agent = {
            "plan": "PlannerAgent",
            "research": "ResearchAgent",
            "code": "CoderAgent",
            "reviewed_code": "ReviewerAgent",
            "execution_result": "CodeExecutionAgent"
        }
        
        if key in key_to_agent:
            agent_name = key_to_agent[key]
            if agent_name in agent_outputs and agent_outputs[agent_name]:
                # Join multiple outputs or get the last one
                outputs = agent_outputs[agent_name]
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    # Join partial responses or return the last (final) one
                    return outputs[-1] if outputs else default
        
        return default
    
    # Get final response (last event's content or last agent's output)
    final_response = "N/A"
    if response:
        last_event = response[-1]
        final_response = extract_text_from_content(last_event.content) if hasattr(last_event, 'content') else "N/A"
    
    # Build result dictionary
    result = {
        "plan": get_value("plan", "N/A"),
        "research": get_value("research", "N/A"),
        "code": get_value("code", "N/A"),
        "reviewed_code": get_value("reviewed_code", "N/A"),
        "execution_result": get_value("execution_result", "N/A"),
        "final_response": final_response
    }
    
    # Debug: Print what we found
    logger.info(f"Extracted state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
    logger.info(f"Agent outputs found: {[k for k, v in agent_outputs.items() if v]}")
    
    return result


# Run the pipeline (only if not running in Streamlit)
if __name__ == "__main__" and not os.getenv("STREAMLIT_RUNNING"):
    print("\n" + "="*70)
    print("Executing Multi-Agent Pipeline...")
    print("="*70)
    
    # Execute the async function
    result = asyncio.run(run_pipeline(user_task))
    
    print("\n--- ✅ EXECUTED CODE ---")
    print(result["execution_result"])

