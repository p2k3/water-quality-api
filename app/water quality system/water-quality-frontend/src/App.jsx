
import { useState } from 'react';
import './App.css';

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

function App() {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
    } catch (err) {
      setResult({ error: 'Failed to fetch prediction.' });
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Water Quality Prediction</h1>
      <form onSubmit={handleSubmit} className="form">
        {Object.keys(initialValues).map((key) => (
          <div key={key} className="form-group">
            <label htmlFor={key}>
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
      {result && (
        <div className="result">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <h2>Prediction Result</h2>
              <p><strong>Water Quality Score:</strong> {result.predictions && result.predictions[0]}</p>
              <p><strong>Status:</strong> {result.explanation && result.explanation.status}</p>
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
