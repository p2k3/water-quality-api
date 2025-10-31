

import { useState } from 'react';
import './App.css';
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
      const response = await fetch('http://localhost:8000/predict', {
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
          <p className="subtitle">Enter your water sample parameters to predict quality and get actionable insights.</p>
        </div>
      </header>
      <form onSubmit={handleSubmit} className="form">
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
      {result && showResult && (
        <div className="result animated-card" style={{ borderLeft: `8px solid ${getStatusColor(result.explanation && result.explanation.status)}` }}>
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <div className="summary-card" style={{ background: getStatusColor(result.explanation && result.explanation.status), color: '#fff', borderRadius: '8px', padding: '1rem', marginBottom: '1rem', boxShadow: '0 2px 8px rgba(42,77,105,0.12)' }}>
                <h2 style={{ margin: 0 }}>{result.explanation && result.explanation.status}</h2>
                <p style={{ margin: 0, fontWeight: 500 }}>Water Quality Status</p>
              </div>
              <p><strong>Water Quality Score:</strong> {result.predictions && result.predictions[0]}</p>
              <p><strong>Main Contributor:</strong> {result.explanation && result.explanation.main_contributor}</p>
              <p><strong>Explanation:</strong> {result.explanation && result.explanation.plain_explanation}</p>
              <h3>Feature Importances</h3>
              <ul>
                {result.explanation && result.explanation.feature_importance &&
                  Object.entries(result.explanation.feature_importance).map(([k, v]) => (
                    <li key={k}><strong>{k}:</strong> {v.toFixed(3)}</li>
                  ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
