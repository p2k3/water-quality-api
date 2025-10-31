
# Water Quality Prediction Frontend

This React app provides a user-friendly interface for predicting water quality using your FastAPI backend.

## Features
- Form for entering water sample parameters (ammonia, bod, dissolved_oxygen, orthophosphate, ph, temperature, nitrogen, nitrate)
- Validation and hints for acceptable ranges
- Sends a POST request to the `/predict` endpoint of your backend
- Displays prediction, water quality status, main contributor, and explanation in a clear format

## Usage
1. Start your FastAPI backend (default: `http://localhost:8000`)
2. Start this React app:
	```
	npm install
	npm run dev
	```
3. Open [http://localhost:5173](http://localhost:5173) in your browser
4. Enter sample parameters and click Predict

## Connecting to a Deployed Backend
If your FastAPI backend is deployed (e.g., on Render), update the API URL in `src/App.jsx`:
```
fetch('http://localhost:8000/predict', ...)
```
to your deployed endpoint, e.g.:
```
fetch('https://your-api.onrender.com/predict', ...)
```

## Customization
- Edit `src/App.jsx` for form logic and API integration
- Edit `src/App.css` for styles

## Project Structure
- `src/App.jsx`: Main app component
- `src/App.css`: Styles

## License
MIT
