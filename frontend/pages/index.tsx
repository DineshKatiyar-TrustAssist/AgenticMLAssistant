import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const SUPERVISED_MODELS = [
  "RandomForestClassifier", "RandomForestRegressor", "LogisticRegression",
  "LinearRegression", "DecisionTreeClassifier", "DecisionTreeRegressor",
  "GradientBoostingClassifier", "GradientBoostingRegressor", "XGBClassifier",
  "XGBRegressor", "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor",
  "NaiveBayes", "AdaBoostClassifier", "AdaBoostRegressor", "ExtraTreesClassifier",
  "ExtraTreesRegressor"
];

const UNSUPERVISED_MODELS = [
  "KMeans", "DBSCAN", "AgglomerativeClustering", "MeanShift"
];
 
interface Result {
  execution_result?: string;
  reviewed_code?: string;
  plan?: string;
  research?: string;
  code?: string;
}

interface ParsedSection {
  section: string;
  icon: string;
  content: string;
  metrics?: { label: string; value: string }[];
  codeBlocks?: { label: string; code: string }[];
}

const parseAndDisplayResults = (resultText: string): ParsedSection[] | null => {
  if (!resultText || resultText === "N/A") return null;

  const sections = resultText.split(/^--- (.+?) ---$/gm);
  if (sections.length <= 1) return null;

  const icons: { [key: string]: string } = {
    "Data Loading": "📊", "EDA": "🔍", "Exploratory": "🔍",
    "Data Cleaning": "🧹", "Preprocessing": "🧹",
    "Model Training": "🤖", "Evaluation": "🤖", "Evaluation Results": "📈"
  };

  const parsed: ParsedSection[] = [];
  let currentSection = '';

  for (let i = 0; i < sections.length; i++) {
    if (i % 2 === 1) {
      currentSection = sections[i].trim();
    } else if (i % 2 === 0 && currentSection && sections[i].trim()) {
      const content = sections[i].trim();
      const iconKey = Object.keys(icons).find(k => currentSection.includes(k)) || '';
      const icon = icons[iconKey] || '';
      
      const section: ParsedSection = {
        section: currentSection,
        icon,
        content
      };

      // Parse Data Loading section
      if (currentSection.includes("Data Loading")) {
        const shapeMatch = content.match(/Dataset Shape: \((\d+), (\d+)\)/);
        if (shapeMatch) {
          section.metrics = [{
            label: "Dataset Shape",
            value: `${shapeMatch[1]} rows × ${shapeMatch[2]} columns`
          }];
        }
        section.codeBlocks = [];
        if (content.includes("First 5 Rows:")) {
          const rowsMatch = content.split("First 5 Rows:")[1]?.split("Data Types:")[0];
          if (rowsMatch) section.codeBlocks.push({ label: "First 5 Rows:", code: rowsMatch.trim() });
        }
        if (content.includes("Data Types:")) {
          const typesMatch = content.split("Data Types:")[1]?.split("---")[0];
          if (typesMatch) section.codeBlocks.push({ label: "Data Types:", code: typesMatch.trim() });
        }
      }

      // Parse EDA section
      if (currentSection.includes("EDA") || currentSection.includes("Exploratory")) {
        section.codeBlocks = [];
        if (content.includes("Summary Statistics")) {
          const statsMatch = content.split("Summary Statistics")[1]?.split("Missing Values")[0];
          if (statsMatch) section.codeBlocks.push({ label: "Summary Statistics:", code: statsMatch.trim() });
        }
        if (content.includes("Missing Values")) {
          const missingMatch = content.split("Missing Values")[1]?.split("---")[0];
          if (missingMatch) section.codeBlocks.push({ label: "Missing Values per Column:", code: missingMatch.trim() });
        }
      }

      // Parse Evaluation Results
      if (currentSection.includes("Evaluation Results")) {
        section.metrics = [];
        const accuracyMatch = content.match(/Accuracy: ([\d.]+)/);
        if (accuracyMatch) {
          section.metrics.push({ label: "Accuracy", value: parseFloat(accuracyMatch[1]).toFixed(4) });
        }
        const rocMatch = content.match(/ROC-AUC Score.*?: ([\d.]+)/);
        if (rocMatch) {
          section.metrics.push({ label: "ROC-AUC Score", value: parseFloat(rocMatch[1]).toFixed(4) });
        }
        section.codeBlocks = [];
        if (content.includes("Classification Report:")) {
          const reportMatch = content.split("Classification Report:")[1]?.split("Confusion Matrix")[0];
          if (reportMatch) section.codeBlocks.push({ label: "Classification Report:", code: reportMatch.trim() });
        }
        if (content.includes("Confusion Matrix:")) {
          const matrixMatch = content.split("Confusion Matrix:")[1];
          if (matrixMatch) section.codeBlocks.push({ label: "Confusion Matrix:", code: matrixMatch.trim() });
        }
      }

      parsed.push(section);
    }
  }

  return parsed.length > 0 ? parsed : null;
};

export default function Home() {
  const [apiKey, setApiKey] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [targetVariable, setTargetVariable] = useState('');
  const [supervisedModel, setSupervisedModel] = useState('None');
  const [unsupervisedModel, setUnsupervisedModel] = useState('None');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResult(null);

    const modelName = supervisedModel !== 'None' ? supervisedModel : 
                     unsupervisedModel !== 'None' ? unsupervisedModel : null;

    if (!file || !targetVariable || !modelName || !apiKey) {
      const missing = [];
      if (!apiKey) missing.push("Google API Key");
      if (!file) missing.push("CSV file");
      if (!targetVariable) missing.push("target variable");
      if (!modelName) missing.push("a model (select from either Supervised or Unsupervised)");
      setError(`⚠️ Please provide: ${missing.join(', ')}`);
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('api_key', apiKey);
    formData.append('target_variable', targetVariable);
    formData.append('model_name', modelName);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${apiUrl}/api/run-pipeline`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getOutputText = () => {
    if (!result?.execution_result || result.execution_result === "N/A") return "";
    
    let outputText = "";
    if (typeof result.execution_result === 'object' && 'message' in result.execution_result) {
      outputText = (result.execution_result as any).message || "";
    } else {
      outputText = String(result.execution_result);
    }
    
    // Extract from sections if embedded
    if (outputText.includes("---")) {
      const match = outputText.match(/--- .+? ---([\s\S]*)/);
      if (match) outputText = match[1];
    }
    
    // Strip asterisks
    return outputText.replace(/\*{1,3}/g, '');
  };

  const parsedResults = result ? parseAndDisplayResults(getOutputText()) : null;

  // Resize handlers
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizing) {
        const newWidth = e.clientX;
        if (newWidth >= 200 && newWidth <= 600) {
          setSidebarWidth(newWidth);
        }
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing]);

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  return (
    <div style={{ 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      display: 'flex',
      minHeight: '100vh',
      backgroundColor: '#f0f2f6'
    }}>
      {/* Sidebar */}
      <div style={{ position: 'relative', display: 'flex' }}>
        <aside 
          ref={sidebarRef}
          style={{
            width: isCollapsed ? '50px' : `${sidebarWidth}px`,
            backgroundColor: 'white',
            padding: '20px',
            boxShadow: '2px 0 5px rgba(0,0,0,0.1)',
            overflowY: 'auto',
            transition: isResizing ? 'none' : 'width 0.2s ease',
            position: 'relative'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: isCollapsed ? '0' : '20px' }}>
            {!isCollapsed && <h2 style={{ margin: 0 }}>📋 Configuration</h2>}
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '20px',
                padding: '5px',
                marginLeft: isCollapsed ? '0' : 'auto',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {isCollapsed ? '▶' : '◀'}
            </button>
          </div>
          
          {!isCollapsed && (
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>
              GOOGLE_API_KEY
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your Google API key"
              style={{
                width: '100%',
                padding: '10px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '14px'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>
              Upload Data/CSV file
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '14px'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>
              Target Column/Prediction Variable
            </label>
            <input
              type="text"
              value={targetVariable}
              onChange={(e) => setTargetVariable(e.target.value)}
              placeholder="e.g., Survived, Class, Price"
              style={{
                width: '100%',
                padding: '10px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '14px'
              }}
            />
          </div>

          <div>
            <h3 style={{ marginTop: 0, marginBottom: '10px', fontSize: '16px' }}>Model Selection</h3>
            
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>
                Supervised ML Model
              </label>
              <select
                value={supervisedModel}
                onChange={(e) => setSupervisedModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px'
                }}
              >
                <option>None</option>
                {SUPERVISED_MODELS.map(m => <option key={m}>{m}</option>)}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 500 }}>
                Unsupervised ML Model
              </label>
              <select
                value={unsupervisedModel}
                onChange={(e) => setUnsupervisedModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '14px'
                }}
              >
                <option>None</option>
                {UNSUPERVISED_MODELS.map(m => <option key={m}>{m}</option>)}
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              backgroundColor: loading ? '#ccc' : '#ff4b4b',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
          >
            {loading && (
              <span style={{
                display: 'inline-block',
                width: '16px',
                height: '16px',
                border: '2px solid rgba(255,255,255,0.3)',
                borderTopColor: 'white',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite'
              }} />
            )}
            {loading ? 'Running...' : '🚀 Run ML Model'}
          </button>
            </form>
          )}
        </aside>
        
        {/* Resize Handle */}
        {!isCollapsed && (
          <div
            onMouseDown={handleResizeStart}
            style={{
              width: '4px',
              backgroundColor: isResizing ? '#999' : '#ddd',
              cursor: 'col-resize',
              position: 'relative',
              flexShrink: 0,
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => {
              if (!isResizing) {
                (e.currentTarget as HTMLElement).style.backgroundColor = '#bbb';
              }
            }}
            onMouseLeave={(e) => {
              if (!isResizing) {
                (e.currentTarget as HTMLElement).style.backgroundColor = '#ddd';
              }
            }}
          />
        )}
      </div>

      {/* Main Content */}
      <main style={{
        flex: 1,
        padding: '40px',
        overflowY: 'auto',
        position: 'relative'
      }}>
        <style dangerouslySetInnerHTML={{__html: `
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}} />
        
        <h1 style={{ marginTop: 0, marginBottom: '10px', fontSize: '32px' }}>
          Agentic AI - ML Assistant
        </h1>
        <p style={{ color: '#666', marginBottom: '30px' }}>
          Upload a CSV file and select an ML model for automated training and evaluation
        </p>

        {loading && (
          <div style={{
            padding: '40px',
            backgroundColor: '#e3f2fd',
            borderRadius: '8px',
            marginBottom: '20px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '20px',
            minHeight: '200px'
          }}>
            <div style={{
              width: '30px',
              height: '30px',
              border: '3px solid rgba(21, 101, 192, 0.2)',
              borderTopColor: '#1565c0',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            <div style={{
              fontSize: '18px',
              fontWeight: 500,
              color: '#1565c0',
              textAlign: 'center'
            }}>
              Running multi-agent pipeline...
              <br />
              <span style={{ fontSize: '14px', fontWeight: 400, color: '#666', marginTop: '8px', display: 'block' }}>
                This may take a few minutes
              </span>
            </div>
          </div>
        )}

        {error && (
          <div style={{
            padding: '15px',
            backgroundColor: '#ffebee',
            color: '#c62828',
            borderRadius: '4px',
            marginBottom: '20px'
          }}>
            {error}
          </div>
        )}

        {!result && !loading && !error && (
          <div style={{
            padding: '20px',
            backgroundColor: '#e3f2fd',
            borderRadius: '4px',
            color: '#1565c0'
          }}>
            👈 Enter your Google API key, upload a CSV file, enter a target variable, and select a model from either Supervised or Unsupervised ML models in the sidebar to get started
          </div>
        )}

        {result && (
          <>
            <div style={{
              padding: '15px',
              backgroundColor: '#e8f5e9',
              color: '#2e7d32',
              borderRadius: '4px',
              marginBottom: '20px',
              fontWeight: 500
            }}>
              ✅ Pipeline completed successfully!
            </div>

            {parsedResults && parsedResults.length > 0 && (
              <div style={{ marginBottom: '30px' }}>
                {parsedResults.map((item, idx) => (
                  <div key={idx} style={{ marginBottom: '30px' }}>
                    <h3 style={{ 
                      fontSize: '20px', 
                      marginBottom: '15px', 
                      fontWeight: 600,
                      backgroundColor: '#e8f5e9',
                      padding: '10px 15px',
                      borderRadius: '4px'
                    }}>
                      {item.icon} {item.section}
                    </h3>
                    
                    {item.metrics && item.metrics.length > 0 && (
                      <div style={{ display: 'flex', gap: '20px', marginBottom: '15px', flexWrap: 'wrap' }}>
                        {item.metrics.map((metric, mIdx) => (
                          <div key={mIdx} style={{
                            padding: '15px 20px',
                            backgroundColor: '#f5f5f5',
                            borderRadius: '4px',
                            minWidth: '150px'
                          }}>
                            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                              {metric.label}
                            </div>
                            <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                              {metric.value}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {item.codeBlocks && item.codeBlocks.length > 0 && (
                      <div style={{ marginBottom: '15px' }}>
                        {item.codeBlocks.map((block, bIdx) => (
                          <div key={bIdx} style={{ marginBottom: '15px' }}>
                            <div style={{ marginBottom: '8px', fontWeight: 500 }}>
                              {block.label}
                            </div>
                            <pre style={{
                              backgroundColor: '#e0e0e0',
                              padding: '15px',
                              borderRadius: '4px',
                              overflow: 'auto',
                              fontSize: '14px',
                              lineHeight: '1.5',
                              margin: 0
                            }}>
                              {block.code}
                            </pre>
                          </div>
                        ))}
                      </div>
                    )}

                    {(!item.codeBlocks || item.codeBlocks.length === 0) && (
                      <div style={{
                        backgroundColor: '#e0e0e0',
                        padding: '15px',
                        borderRadius: '4px',
                        whiteSpace: 'pre-wrap',
                        fontSize: '14px',
                        lineHeight: '1.5'
                      }}>
                        {item.content}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {parsedResults === null && result?.execution_result && result.execution_result !== "N/A" && (
              <div style={{
                backgroundColor: '#e0e0e0',
                padding: '15px',
                borderRadius: '4px',
                maxHeight: '400px',
                overflow: 'auto'
              }}>
                <div style={{ 
                  margin: 0, 
                  fontSize: '14px', 
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'system-ui, -apple-system, sans-serif'
                }}>
                  {getOutputText()}
                </div>
              </div>
            )}

            {result.reviewed_code && result.reviewed_code !== "N/A" && (
              <details style={{ marginBottom: '20px' }}>
                <summary style={{
                  cursor: 'pointer',
                  padding: '10px',
                  backgroundColor: '#81c784',
                  borderRadius: '4px',
                  fontWeight: 500,
                  color: 'white'
                }}>
                  📝 Generated Code
                </summary>
                <pre style={{
                  backgroundColor: '#e0e0e0',
                  padding: '15px',
                  borderRadius: '4px',
                  overflow: 'auto',
                  marginTop: '10px',
                  fontSize: '14px',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  margin: 0
                }}>
                  {result.reviewed_code}
                </pre>
              </details>
            )}

            {result.plan && result.plan !== "N/A" && (
              <details style={{ marginBottom: '20px' }}>
                <summary style={{
                  cursor: 'pointer',
                  padding: '10px',
                  backgroundColor: '#81c784',
                  borderRadius: '4px',
                  fontWeight: 500,
                  color: 'white'
                }}>
                  📋 Execution Plan
                </summary>
                <div style={{
                  backgroundColor: '#e0e0e0',
                  padding: '15px',
                  borderRadius: '4px',
                  marginTop: '10px',
                  whiteSpace: 'pre-wrap',
                  fontSize: '14px',
                  fontFamily: 'system-ui, -apple-system, sans-serif'
                }}>
                  {result.plan}
                </div>
              </details>
            )}
          </>
        )}
      </main>
    </div>
  );
}
