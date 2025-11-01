// Field explanations for dashboard clarity
const fieldExplanations = {
  classification: "Compliant: The water sample meets all regulatory standards (DEAS12:2018).",
  forecast: "Water Quality Status: The predicted status based on your sample; 'Safe' means low risk for pollution or health hazards.",
  risk_score: "Risk Score: Probability (0–1) of non-compliance; 0 means no risk detected.",
  predictions: "Water Quality Score: Model’s overall score for water safety (higher is better; scale may vary by model).",
  parameter_breakdown: "Parameter Breakdown: Lists any parameters that are out of compliance; 'All parameters within compliance limits' means everything is within safe ranges.",
  pollutant_probabilities: "Pollutant Probabilities: Model’s estimated probability for each pollutant type; higher values indicate greater likelihood of that pollutant.",
  feature_importances: "Feature Importances: Shows which input features most influenced the prediction; 0.000 means none of the features had a strong impact for this sample.",
  plain_explanation: "Explanation: Plain-language summary of why the sample is classified as it is.",
  audit: "Audit Info: Timestamp and model version for traceability."
};

const pollutantTypeLabels = [
  "Bacterial (e.g., ammonia, BOD, low oxygen)",
  "Chemical (e.g., pH, nitrogen, nitrate)",
  "Organic (e.g., BOD, ammonia, low oxygen)",
  "Agricultural (e.g., nitrate, orthophosphate, nitrogen)"
];


import { useState } from 'react';
import './App.css';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import ScienceIcon from '@mui/icons-material/Science';
import OpacityIcon from '@mui/icons-material/Opacity';
import BiotechIcon from '@mui/icons-material/Biotech';
import ThermostatIcon from '@mui/icons-material/Thermostat';
import DeviceThermostatIcon from '@mui/icons-material/DeviceThermostat';
import FunctionsIcon from '@mui/icons-material/Functions';
import logo from './assets/react.svg'; // Replace with your own logo if desired

const initialValues = {
  ammonia: '',
  bod: '',
  dissolved_oxygen: '',
  orthophosphate: '',
  ph: '',
  temperature: '',
  nitrogen: '',
  nitrate: ''
};

const ranges = {
  ammonia: '0.001–0.5 mg/L',
  bod: '0.001–5.0 mg/L',
  dissolved_oxygen: '6.0+ mg/L',
  orthophosphate: '0.001–0.1 mg/L',
  ph: '6.5–8.5',
  temperature: '0.001–30°C',
  nitrogen: '0.001–10.0 mg/L',
  nitrate: '0.001–10.0 mg/L'
};

const icons = {
  ammonia: <WaterDropIcon style={{ color: '#2a4d69' }} />,
  bod: <ScienceIcon style={{ color: '#4b6cb7' }} />,
  dissolved_oxygen: <OpacityIcon style={{ color: '#1976d2' }} />,
  orthophosphate: <BiotechIcon style={{ color: '#388e3c' }} />,
  ph: <FunctionsIcon style={{ color: '#7b1fa2' }} />,
  temperature: <DeviceThermostatIcon style={{ color: '#f9a825' }} />,
  nitrogen: <BiotechIcon style={{ color: '#0288d1' }} />,
  nitrate: <BiotechIcon style={{ color: '#0288d1' }} />,
};

const tooltips = {
  ammonia: 'Ammonia: Indicates pollution from sewage or fertilizer runoff.',
  bod: 'BOD: Measures organic pollution in water.',
  dissolved_oxygen: 'Dissolved Oxygen: Essential for aquatic life.',
  orthophosphate: 'Orthophosphate: Linked to agricultural runoff.',
  ph: 'pH: Indicates acidity/alkalinity.',
  temperature: 'Temperature: Affects water chemistry and biology.',
  nitrogen: 'Nitrogen: Can indicate fertilizer or waste pollution.',
  nitrate: 'Nitrate: Often from agricultural sources.'
};


function getStatusColor(status) {
  if (status === 'Safe') return '#2a4d69';
  if (status === 'Moderate') return '#f9a825';
  if (status === 'Unsafe') return '#d7263d';
  return '#4b6cb7';
}

function App() {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [showExplanationsCard, setShowExplanationsCard] = useState(false);

  const validate = () => {
    const newErrors = {};
    if (values.ammonia < 0.001 || values.ammonia > 0.5) newErrors.ammonia = 'Must be 0.001–0.5';
    if (values.bod < 0.001 || values.bod > 5.0) newErrors.bod = 'Must be 0.001–5.0';
    if (values.dissolved_oxygen < 6.0) newErrors.dissolved_oxygen = 'Must be 6.0+';
    if (values.orthophosphate < 0.001 || values.orthophosphate > 0.1) newErrors.orthophosphate = 'Must be 0.001–0.1';
    if (values.ph < 6.5 || values.ph > 8.5) newErrors.ph = 'Must be 6.5–8.5';
    if (values.temperature < 0.001 || values.temperature > 30) newErrors.temperature = 'Must be 0.001–30';
    if (values.nitrogen < 0.001 || values.nitrogen > 10.0) newErrors.nitrogen = 'Must be 0.001–10.0';
    if (values.nitrate < 0.001 || values.nitrate > 10.0) newErrors.nitrate = 'Must be 0.001–10.0';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    setValues({ ...values, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    setResult(null);
    setShowResult(false);
    try {
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: [{
          ammonia: parseFloat(values.ammonia),
          bod: parseFloat(values.bod),
          dissolved_oxygen: parseFloat(values.dissolved_oxygen),
          orthophosphate: parseFloat(values.orthophosphate),
          ph: parseFloat(values.ph),
          temperature: parseFloat(values.temperature),
          nitrogen: parseFloat(values.nitrogen),
          nitrate: parseFloat(values.nitrate)
        }] })
      });
      const data = await response.json();
      setResult(data);
      setTimeout(() => setShowResult(true), 200);
    } catch (err) {
      setResult({ error: 'Failed to fetch prediction.' });
      setShowResult(true);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <header className="header">
        <img src={logo} alt="Water Quality Logo" className="logo-header" />
        <div>
          <h1>Water Quality Predictor</h1>
          <p className="subtitle">Enter your water sample parameters below to get a clear, actionable dashboard of water quality predictions and explanations.</p>
        </div>
      </header>
      <form onSubmit={handleSubmit} className="form">
        <h2>Enter Water Sample Parameters</h2>
        {Object.keys(initialValues).map((key) => (
          <div key={key} className="form-group">
            <label htmlFor={key} title={tooltips[key]} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'help' }}>
              <span style={{ fontSize: '1.3rem' }}>{icons[key]}</span>
              {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} ({ranges[key]})
            </label>
            <input
              type="number"
              step="any"
              name={key}
              id={key}
              value={values[key]}
              onChange={handleChange}
              required
            />
            {errors[key] && <span className="error">{errors[key]}</span>}
          </div>
        ))}
        <button type="submit" disabled={loading}>Predict</button>
      </form>
      {loading && <p>Loading...</p>}
      {/* Add extra spacing between Predict button and results */}
      <div style={{ height: showResult ? '2.5rem' : '0' }}></div>
      {result && showResult && (
        <div className="dashboard animated-card">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <div className="summary-card" style={{ background: getStatusColor(result.forecast), color: '#fff', borderRadius: '16px', padding: '1.2rem 1rem', marginBottom: '1.2rem', boxShadow: '0 2px 12px rgba(42,77,105,0.15)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
                  <div>
                    <h2 style={{ margin: 0 }}>{result.classification}</h2>
                    <div style={{ marginTop: '6px', fontWeight: 600 }}>
                      <span style={{ fontSize: '0.98rem' }}>Status: </span>
                      <span style={{ color: getStatusColor(result.forecast), fontWeight: 700, fontSize: '1rem' }}>{result.forecast}</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.95rem', fontWeight: 700 }}>Score</div>
                      <div style={{ fontSize: '1.05rem' }}>{(result.predictions && result.predictions[0]) ? Number(result.predictions[0]).toFixed(3) : '—'}</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.95rem', fontWeight: 700 }}>Risk</div>
                      <div style={{ fontSize: '1.05rem' }}>{typeof result.risk_score === 'number' ? Number(result.risk_score).toFixed(3) : (result.risk_score || '—')}</div>
                    </div>
                    <button className="info-button" onClick={() => setShowExplanationsCard(!showExplanationsCard)} aria-label="Toggle explanations" title={showExplanationsCard ? 'Hide details' : 'Show details'}>
                      <InfoOutlinedIcon style={{ color: '#fff' }} />
                    </button>
                  </div>
                </div>
                {showExplanationsCard && (
                  <div className="summary-explanations" style={{ marginTop: '10px', background: 'rgba(255,255,255,0.08)', padding: '10px', borderRadius: '8px', color: 'rgba(255,255,255,0.95)' }}>
                    <p style={{ margin: '6px 0' }}><strong>Classification:</strong> {fieldExplanations.classification}</p>
                    <p style={{ margin: '6px 0' }}><strong>Forecast:</strong> {fieldExplanations.forecast}</p>
                    <p style={{ margin: '6px 0' }}><strong>Risk Score:</strong> {fieldExplanations.risk_score}</p>
                    <p style={{ margin: '6px 0' }}><strong>Water Quality Score:</strong> {fieldExplanations.predictions}</p>
                  </div>
                )}
              </div>
              {/* Compact results box: displays core numeric outputs before the full explanation */}
              <div className="results-box" style={{ background: '#ffffff', borderRadius: '12px', padding: '1rem', marginBottom: '1rem', boxShadow: '0 2px 8px rgba(0,0,0,0.06)', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <div style={{ flex: 1 }}>
                  <h4 style={{ margin: 0, color: '#2a4d69' }}>Quick Results</h4>
                  <p style={{ margin: '0.35rem 0 0 0' }}><strong>Water Quality Score:</strong> {(result.predictions && result.predictions[0]) ? Number(result.predictions[0]).toFixed(3) : '—'}</p>
                  <p style={{ margin: '0.2rem 0 0 0' }}><strong>Forecast:</strong> <span style={{ color: getStatusColor(result.forecast), fontWeight: 700 }}>{result.forecast || '—'}</span></p>
                  <p style={{ margin: '0.2rem 0 0 0' }}><strong>Risk Score:</strong> {typeof result.risk_score === 'number' ? result.risk_score.toFixed(3) : (result.risk_score || '—')}</p>
                </div>
                <div style={{ width: '240px', textAlign: 'left' }}>
                  <h5 style={{ margin: '0 0 0.35rem 0' }}>Top Pollutant</h5>
                  {result.pollutant_probabilities && result.pollutant_probabilities.length > 0 ? (() => {
                    const probs = result.pollutant_probabilities[0];
                    const maxIdx = probs.indexOf(Math.max(...probs));
                    const label = pollutantTypeLabels[maxIdx] || `Type ${maxIdx + 1}`;
                    return (
                      <div>
                        <p style={{ margin: 0 }}><strong>{label}</strong></p>
                        <p style={{ margin: '0.25rem 0 0 0' }}>{(probs[maxIdx] * 100).toFixed(1)}% probability</p>
                      </div>
                    );
                  })() : <p style={{ margin: 0 }}>No pollutant data</p>}
                </div>
              </div>

              <div className="explanation-card" style={{ background: '#fff', borderRadius: '12px', boxShadow: '0 2px 8px rgba(42,77,105,0.09)', padding: '1.2rem 1rem', marginBottom: '1.2rem', borderLeft: '6px solid #4b6cb7' }}>
                <h3 style={{ marginTop: 0, color: '#2a4d69' }}>Explanation</h3>
                <small className="field-explanation">{fieldExplanations.plain_explanation}</small>
                <p style={{ fontSize: '1.12rem', margin: '0.7rem 0' }}>{result.explainability && result.explainability.plain_explanation}</p>
              </div>
              <div className="dashboard-section">
                <h3>Parameter Breakdown</h3>
                <small className="field-explanation">{fieldExplanations.parameter_breakdown}</small>
                {result.parameter_breakdown && result.parameter_breakdown.length > 0 ? (
                  <ul>
                    {result.parameter_breakdown.map((item, idx) => (
                      <li key={idx}>
                        <strong>{item.parameter}:</strong> {item.value} ({item.reason}, Limit: {item.limit})
                      </li>
                    ))}
                  </ul>
                ) : <p>All parameters within compliance limits.</p>}
              </div>
              <div className="dashboard-section">
                <h3>Pollutant Probabilities</h3>
                <small className="field-explanation">{fieldExplanations.pollutant_probabilities}</small>
                {result.pollutant_probabilities && result.pollutant_probabilities.length > 0 && (
                  <ul>
                    {result.pollutant_probabilities[0].map((prob, idx) => (
                      <li key={idx}><strong>{pollutantTypeLabels[idx] || `Type ${idx + 1}`}:</strong> {prob.toFixed(3)}</li>
                    ))}
                  </ul>
                )}
              </div>
              <div className="dashboard-section">
                <h3>Feature Importances</h3>
                <small className="field-explanation">{fieldExplanations.feature_importances}</small>
                <ul>
                  {result.explainability && result.explainability.feature_scores &&
                    Object.entries(result.explainability.feature_scores).map(([k, v]) => (
                      <li key={k}><strong>{k}:</strong> {v.toFixed(3)}</li>
                    ))}
                </ul>
              </div>
              <div className="audit-info">
                <h3>Audit Info</h3>
                <small className="field-explanation">{fieldExplanations.audit}</small>
                <p><strong>Timestamp:</strong> {result.audit && result.audit.timestamp}</p>
                <p><strong>Model Version:</strong> {result.audit && result.audit.model_version}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
