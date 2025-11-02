// IMPROVED Water Quality Frontend - Efficiently utilizes backend with enhanced UX
// Changes:
// 1. Relaxed validation to allow testing unsafe water samples
// 2. Updated to DEAS 21:2018 standard
// 3. Sorted and color-coded SHAP feature importances
// 4. Visual progress bars for pollutant probabilities
// 5. Severity badges for parameter violations
// 6. Better organization and readability

import { useState } from 'react';
import './App.css';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import ScienceIcon from '@mui/icons-material/Science';
import OpacityIcon from '@mui/icons-material/Opacity';
import BiotechIcon from '@mui/icons-material/Biotech';
import DeviceThermostatIcon from '@mui/icons-material/DeviceThermostat';
import FunctionsIcon from '@mui/icons-material/Functions';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import logo from './assets/react.svg';

// Field explanations - UPDATED TO DEAS 21:2018
const fieldExplanations = {
  classification: "Classification based on DEAS 21:2018 (EAS 12:2018) water quality standards for East Africa.",
  forecast: "Water Quality Forecast: Overall safety assessment - 'Safe' (low risk), 'Moderate' (monitor closely), or 'Unsafe' (immediate action required).",
  risk_score: "Risk Score: Probability (0–1) of regulatory non-compliance; calculated from model predictions and parameter violations.",
  predictions: "Water Quality Score: Model's sigmoid-transformed prediction (0-1 scale); higher values indicate better water quality.",
  parameter_breakdown: "Parameter Breakdown: DEAS 21:2018 compliance check for each water quality parameter. Shows severity and deviation percentage.",
  pollutant_probabilities: "Pollutant Probabilities: Model's multi-class prediction for pollution source type. Helps identify root causes.",
  feature_importances: "Feature Importances (SHAP): Explainable AI showing which parameters drove the prediction. Positive values push toward 'unsafe', negative toward 'safe'. Sorted by absolute impact.",
  plain_explanation: "Analysis Summary: Plain-language explanation of water quality assessment based on SHAP analysis and DEAS 21:2018 standards.",
  audit: "Audit Trail: Timestamp and model version for regulatory compliance and reproducibility."
};

const pollutantTypeLabels = [
  "Bacterial (e.g., ammonia, BOD, low oxygen)",
  "Chemical (e.g., pH, nitrogen, nitrate)",
  "Organic (e.g., BOD, ammonia, low oxygen)",
  "Agricultural (e.g., nitrate, orthophosphate, nitrogen)"
];

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

// DEAS 21:2018 COMPLIANCE ranges (for display only)
const deasRanges = {
  ammonia: { min: 0.001, max: 0.5, unit: 'mg/L', standard: 'DEAS 21:2018' },
  bod: { min: 0.001, max: 5.0, unit: 'mg/L', standard: 'Typical limit' },
  dissolved_oxygen: { min: 6.0, max: Infinity, unit: 'mg/L', standard: 'DEAS 21:2018' },
  orthophosphate: { min: 0.001, max: 0.1, unit: 'mg/L', standard: 'Typical limit' },
  ph: { min: 6.5, max: 8.5, unit: '', standard: 'DEAS 21:2018' },
  temperature: { min: 0.001, max: 30, unit: '°C', standard: 'Typical range' },
  nitrogen: { min: 0.001, max: 10.0, unit: 'mg/L', standard: 'Typical limit' },
  nitrate: { min: 0.001, max: 50.0, unit: 'mg/L', standard: 'DEAS 21:2018' }
};

const icons = {
  ammonia: <WaterDropIcon style={{ color: '#2a4d69' }} />,
  bod: <ScienceIcon style={{ color: '#4b6cb7' }} />,
  dissolved_oxygen: <OpacityIcon style={{ color: '#1976d2' }} />,
  orthophosphate: <BiotechIcon style={{ color: '#388e3c' }} />,
  ph: <FunctionsIcon style={{ color: '#7b1fa2' }} />,
  temperature: <DeviceThermostatIcon style={{ color: '#f9a825' }} />,
  nitrogen: <BiotechIcon style={{ color: '#0288d1' }} />,
  nitrate: <BiotechIcon style={{ color: '#0288d1' }} />
};

const tooltips = {
  ammonia: 'Ammonia (NH₃): Indicates sewage or fertilizer runoff. DEAS limit: ≤0.5 mg/L',
  bod: 'BOD: Biological Oxygen Demand - measures organic pollution load.',
  dissolved_oxygen: 'Dissolved Oxygen: Critical for aquatic life. DEAS minimum: ≥6.0 mg/L',
  orthophosphate: 'Orthophosphate (PO₄³⁻): Agricultural runoff indicator.',
  ph: 'pH: Acidity/alkalinity balance. DEAS range: 6.5–8.5',
  temperature: 'Temperature: Affects water chemistry and biological activity.',
  nitrogen: 'Total Nitrogen: Multiple pollution sources indicator.',
  nitrate: 'Nitrate (NO₃⁻): Agricultural fertilizer runoff. DEAS limit: ≤50 mg/L'
};

function getStatusColor(cardColor, forecast) {
  // Use backend card_color if available, otherwise fall back to forecast
  if (cardColor === 'success') return '#2e7d32';  // Green
  if (cardColor === 'warning') return '#f57c00';  // Orange
  if (cardColor === 'danger') return '#c62828';   // Red
  
  // Fallback to forecast-based color
  if (forecast === 'Safe') return '#2e7d32';
  if (forecast === 'Moderate') return '#f57c00';
  if (forecast === 'Unsafe') return '#c62828';
  return '#4b6cb7';  // Default blue
}

function getSeverityColor(severity) {
  if (severity === 'CRITICAL') return '#c62828';
  if (severity === 'MODERATE') return '#f57c00';
  return '#757575';
}

function App() {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [showExplanationsCard, setShowExplanationsCard] = useState(false);

  // RELAXED VALIDATION - Only check for required fields, not ranges
  // This allows testing unsafe/contaminated water samples
  const validate = () => {
    const newErrors = {};
    Object.keys(initialValues).forEach(key => {
      if (!values[key] || values[key] === '') {
        newErrors[key] = 'Required field';
      }
    });
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    setValues({ ...values, [e.target.name]: e.target.value });
    // Clear error for this field
    if (errors[e.target.name]) {
      setErrors({ ...errors, [e.target.name]: undefined });
    }
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
        body: JSON.stringify({ 
          features: [{
            ammonia: parseFloat(values.ammonia),
            bod: parseFloat(values.bod),
            dissolved_oxygen: parseFloat(values.dissolved_oxygen),
            orthophosphate: parseFloat(values.orthophosphate),
            ph: parseFloat(values.ph),
            temperature: parseFloat(values.temperature),
            nitrogen: parseFloat(values.nitrogen),
            nitrate: parseFloat(values.nitrate)
          }] 
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'API request failed');
      }
      
      const data = await response.json();
      setResult(data);
      setTimeout(() => setShowResult(true), 200);
    } catch (err) {
      console.error('Prediction error:', err);
      setResult({ error: `Failed to fetch prediction: ${err.message}` });
      setShowResult(true);
    }
    
    setLoading(false);
  };

  // Helper: Render feature importances with sorting and color coding
  const renderFeatureImportances = () => {
    if (!result.explainability || !result.explainability.feature_scores) {
      return <p>No feature importance data available</p>;
    }

    const scores = result.explainability.feature_scores;
    // Sort by absolute value (most important first)
    const sorted = Object.entries(scores).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    const maxAbsValue = Math.max(...sorted.map(([, v]) => Math.abs(v)));
    
    return (
      <div style={{ marginTop: '0.5rem' }}>
        {sorted.map(([feature, value]) => {
          const absValue = Math.abs(value);
          const percentage = maxAbsValue > 0 ? (absValue / maxAbsValue) * 100 : 0;
          const isPositive = value > 0;
          const color = isPositive ? '#c62828' : '#2e7d32';
          const bgColor = isPositive ? '#ffebee' : '#e8f5e9';
          
          return (
            <div key={feature} style={{ 
              marginBottom: '0.75rem',
              padding: '0.5rem',
              background: bgColor,
              borderRadius: '6px',
              borderLeft: `4px solid ${color}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                <strong style={{ color: '#2a4d69' }}>
                  {feature.replace('_', ' ')}
                </strong>
                <span style={{ 
                  color: color, 
                  fontWeight: 'bold',
                  fontSize: '0.95rem'
                }}>
                  {value > 0 ? '+' : ''}{value.toFixed(4)}
                </span>
              </div>
              <div style={{ 
                width: '100%', 
                height: '6px', 
                background: '#e0e0e0', 
                borderRadius: '3px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: `${percentage}%`, 
                  height: '100%', 
                  background: color,
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
              <small style={{ color: '#666', fontSize: '0.8rem' }}>
                {isPositive ? '↑ Pushes toward unsafe' : '↓ Pushes toward safe'}
              </small>
            </div>
          );
        })}
        {result.explainability.main_feature && (
          <div style={{ 
            marginTop: '1rem', 
            padding: '0.75rem', 
            background: '#fff3e0', 
            borderRadius: '8px',
            borderLeft: '4px solid #f57c00'
          }}>
            <strong>🎯 Key Driver:</strong> {result.explainability.main_feature.replace('_', ' ')} 
            <span style={{ marginLeft: '0.5rem', color: '#f57c00', fontWeight: 'bold' }}>
              (SHAP: {result.explainability.main_feature_value?.toFixed(4)})
            </span>
          </div>
        )}
      </div>
    );
  };

  // Helper: Render pollutant probabilities with progress bars
  const renderPollutantProbabilities = () => {
    if (!result.pollutant_probabilities || result.pollutant_probabilities.length === 0) {
      return <p>No pollutant probability data available</p>;
    }

    const probs = result.pollutant_probabilities[0];
    const maxProb = Math.max(...probs);
    
    return (
      <div style={{ marginTop: '0.5rem' }}>
        {probs.map((prob, idx) => {
          const isMax = prob === maxProb;
          const percentage = (prob * 100).toFixed(1);
          const barColor = isMax ? '#2e7d32' : '#90a4ae';
          
          return (
            <div key={idx} style={{ 
              marginBottom: '1rem',
              padding: '0.75rem',
              background: isMax ? '#e8f5e9' : '#f5f5f5',
              borderRadius: '8px',
              borderLeft: isMax ? '4px solid #2e7d32' : '4px solid #cfd8dc'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <strong style={{ color: '#2a4d69', fontSize: '0.95rem' }}>
                  {isMax && '🏆 '}{pollutantTypeLabels[idx]}
                </strong>
                <span style={{ 
                  fontWeight: 'bold', 
                  color: barColor,
                  fontSize: '1rem'
                }}>
                  {percentage}%
                </span>
              </div>
              <div style={{ 
                width: '100%', 
                height: '8px', 
                background: '#e0e0e0', 
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: `${percentage}%`, 
                  height: '100%', 
                  background: barColor,
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Helper: Render parameter breakdown with severity badges
  const renderParameterBreakdown = () => {
    if (!result.parameter_breakdown || result.parameter_breakdown.length === 0) {
      return (
        <div style={{ 
          padding: '1rem', 
          background: '#e8f5e9', 
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <CheckCircleIcon style={{ color: '#2e7d32' }} />
          <span style={{ color: '#2e7d32', fontWeight: 600 }}>
            All parameters within DEAS 21:2018 compliance limits ✓
          </span>
        </div>
      );
    }

    return (
      <div style={{ marginTop: '0.5rem' }}>
        {result.parameter_breakdown.map((item, idx) => {
          const severity = item.severity || 'MODERATE';
          const severityColor = getSeverityColor(severity);
          
          return (
            <div key={idx} style={{ 
              marginBottom: '0.75rem',
              padding: '0.75rem',
              background: severity === 'CRITICAL' ? '#ffebee' : '#fff3e0',
              borderRadius: '8px',
              borderLeft: `4px solid ${severityColor}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                    <WarningIcon style={{ color: severityColor, fontSize: '1.2rem' }} />
                    <strong style={{ color: '#2a4d69' }}>
                      {item.parameter.replace('_', ' ').toUpperCase()}
                    </strong>
                    <span style={{ 
                      padding: '2px 8px', 
                      background: severityColor, 
                      color: '#fff', 
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      fontWeight: 'bold'
                    }}>
                      {severity}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.9rem', color: '#555' }}>
                    <strong>Value:</strong> {Number(item.value).toFixed(3)} {deasRanges[item.parameter]?.unit || ''}
                  </div>
                  <div style={{ fontSize: '0.9rem', color: '#555' }}>
                    <strong>Limit:</strong> {item.limit} ({deasRanges[item.parameter]?.standard || 'DEAS 21:2018'})
                  </div>
                  {item.deviation_percent && (
                    <div style={{ fontSize: '0.85rem', color: severityColor, fontWeight: 600, marginTop: '0.25rem' }}>
                      ⚠ {item.deviation_percent.toFixed(1)}% deviation from limit
                    </div>
                  )}
                </div>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#666', fontStyle: 'italic' }}>
                {item.reason}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="container">
      <header className="header">
        <img src={logo} alt="Water Quality Logo" className="logo-header" />
        <div>
          <h1>Water Quality Predictor</h1>
          <p className="subtitle">
            AI-powered water quality analysis using SymGAT + SHAP explainability. 
            Based on DEAS 21:2018 (EAS 12:2018) standards for East Africa.
          </p>
        </div>
      </header>
      
      <form onSubmit={handleSubmit} className="form">
        <h2>Enter Water Sample Parameters</h2>
        {Object.keys(initialValues).map((key) => {
          const range = deasRanges[key];
          const rangeText = range.max === Infinity 
            ? `≥${range.min} ${range.unit}` 
            : `${range.min}–${range.max} ${range.unit}`;
          
          return (
            <div key={key} className="form-group">
              <label htmlFor={key} title={tooltips[key]} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem', 
                cursor: 'help' 
              }}>
                <span style={{ fontSize: '1.3rem' }}>{icons[key]}</span>
                {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                <span style={{ fontSize: '0.85rem', color: '#757575' }}>
                  ({rangeText})
                </span>
              </label>
              <input
                type="number"
                step="any"
                name={key}
                id={key}
                value={values[key]}
                onChange={handleChange}
                required
                placeholder={`Enter ${key.replace('_', ' ')}...`}
              />
              {errors[key] && <span className="error">{errors[key]}</span>}
            </div>
          );
        })}
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Predict Water Quality'}
        </button>
      </form>
      
      {loading && (
        <div style={{ textAlign: 'center', marginTop: '2rem', color: '#4b6cb7' }}>
          <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>
            🔬 Analyzing water sample with SymGAT model...
          </p>
          <p style={{ fontSize: '0.9rem', color: '#757575' }}>
            Computing SHAP feature importances and DEAS 21:2018 compliance...
          </p>
        </div>
      )}
      
      <div style={{ height: showResult ? '2.5rem' : '0' }}></div>
      
      {result && showResult && (
        <div className="dashboard animated-card">
          {result.error ? (
            <div style={{ 
              padding: '2rem', 
              background: '#ffebee', 
              borderRadius: '12px',
              borderLeft: '6px solid #c62828'
            }}>
              <p className="error" style={{ fontSize: '1.1rem', fontWeight: 600 }}>
                ❌ {result.error}
              </p>
              <p style={{ marginTop: '1rem', color: '#666' }}>
                Please check your input values and try again. If the problem persists, 
                the backend service may be unavailable.
              </p>
            </div>
          ) : (
            <>
              {/* Summary Card */}
              <div className="summary-card" style={{ 
                background: getStatusColor(result.card_color, result.forecast), 
                color: '#fff', 
                borderRadius: '16px', 
                padding: '1.5rem', 
                marginBottom: '1.5rem', 
                boxShadow: '0 4px 20px rgba(0,0,0,0.15)' 
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
                  <div>
                    <h2 style={{ margin: 0, fontSize: '1.5rem' }}>
                      {result.classification}
                    </h2>
                    <div style={{ marginTop: '0.75rem', fontSize: '1.1rem', fontWeight: 600 }}>
                      Status: <span style={{ fontWeight: 700 }}>{result.forecast}</span>
                      {result.confidence && (
                        <span style={{ 
                          marginLeft: '0.75rem',
                          padding: '2px 8px',
                          background: 'rgba(255,255,255,0.25)',
                          borderRadius: '4px',
                          fontSize: '0.85rem',
                          fontWeight: 500
                        }}>
                          {result.confidence} Confidence
                        </span>
                      )}
                    </div>
                    <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', opacity: 0.9 }}>
                      DEAS 21:2018 Assessment
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>Quality Score</div>
                      <div style={{ fontSize: '2rem', fontWeight: 700 }}>
                        {(result.predictions?.[0])?.toFixed(3) || '—'}
                      </div>
                      <div style={{ fontSize: '0.75rem', opacity: 0.8, marginTop: '0.25rem' }}>
                        Raw: {(result.raw_predictions?.[0])?.toFixed(3) || '—'}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.85rem', opacity: 0.9 }}>Risk Score</div>
                      <div style={{ fontSize: '2rem', fontWeight: 700 }}>
                        {typeof result.risk_score === 'number' ? result.risk_score.toFixed(3) : '—'}
                      </div>
                    </div>
                    <button 
                      className="info-button" 
                      onClick={() => setShowExplanationsCard(!showExplanationsCard)} 
                      aria-label="Toggle explanations" 
                      title={showExplanationsCard ? 'Hide details' : 'Show metric explanations'}
                    >
                      <InfoOutlinedIcon style={{ color: '#fff', fontSize: '1.5rem' }} />
                    </button>
                  </div>
                </div>
                {showExplanationsCard && (
                  <div className="summary-explanations" style={{ 
                    marginTop: '1rem', 
                    background: 'rgba(255,255,255,0.15)', 
                    padding: '1rem', 
                    borderRadius: '8px' 
                  }}>
                    <p style={{ margin: '0.5rem 0' }}><strong>Classification:</strong> {fieldExplanations.classification}</p>
                    <p style={{ margin: '0.5rem 0' }}><strong>Forecast:</strong> {fieldExplanations.forecast}</p>
                    <p style={{ margin: '0.5rem 0' }}><strong>Risk Score:</strong> {fieldExplanations.risk_score}</p>
                    <p style={{ margin: '0.5rem 0' }}><strong>Quality Score:</strong> {fieldExplanations.predictions}</p>
                  </div>
                )}
              </div>

              {/* Plain Explanation Card */}
              <div className="explanation-card" style={{ 
                background: '#fff', 
                borderRadius: '12px', 
                boxShadow: '0 2px 12px rgba(0,0,0,0.08)', 
                padding: '1.5rem', 
                marginBottom: '1.5rem', 
                borderLeft: '6px solid #4b6cb7' 
              }}>
                <h3 style={{ marginTop: 0, color: '#2a4d69', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <InfoOutlinedIcon />
                  Analysis Summary
                </h3>
                <small className="field-explanation" style={{ display: 'block', marginBottom: '0.75rem', color: '#757575' }}>
                  {fieldExplanations.plain_explanation}
                </small>
                <p style={{ fontSize: '1.05rem', lineHeight: '1.6', margin: 0, color: '#333' }}>
                  {result.explainability?.plain_explanation || 'No explanation available.'}
                </p>
              </div>

              {/* Parameter Breakdown */}
              <div className="dashboard-section">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <WarningIcon style={{ color: '#f57c00' }} />
                  Parameter Breakdown (DEAS 21:2018)
                </h3>
                <small className="field-explanation" style={{ display: 'block', marginBottom: '0.75rem' }}>
                  {fieldExplanations.parameter_breakdown}
                </small>
                {renderParameterBreakdown()}
              </div>

              {/* Pollutant Probabilities */}
              <div className="dashboard-section">
                <h3>Pollutant Source Probabilities</h3>
                <small className="field-explanation" style={{ display: 'block', marginBottom: '0.75rem' }}>
                  {fieldExplanations.pollutant_probabilities}
                </small>
                {renderPollutantProbabilities()}
              </div>

              {/* Feature Importances (SHAP) */}
              <div className="dashboard-section">
                <h3>Feature Importances (SHAP Explainability)</h3>
                <small className="field-explanation" style={{ display: 'block', marginBottom: '0.75rem' }}>
                  {fieldExplanations.feature_importances}
                </small>
                {renderFeatureImportances()}
              </div>

              {/* Audit Info */}
              <div className="audit-info" style={{ 
                marginTop: '1.5rem', 
                padding: '1rem', 
                background: '#f5f5f5', 
                borderRadius: '8px',
                fontSize: '0.9rem'
              }}>
                <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem' }}>Audit Trail</h3>
                <small className="field-explanation" style={{ display: 'block', marginBottom: '0.5rem' }}>
                  {fieldExplanations.audit}
                </small>
                <p style={{ margin: '0.25rem 0' }}>
                  <strong>Timestamp:</strong> {result.audit?.timestamp ? new Date(result.audit.timestamp).toLocaleString() : '—'}
                </p>
                <p style={{ margin: '0.25rem 0' }}>
                  <strong>Model Version:</strong> {result.audit?.model_version || '—'}
                </p>
                <p style={{ margin: '0.25rem 0' }}>
                  <strong>Standard:</strong> DEAS 21:2018 (EAS 12:2018)
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
