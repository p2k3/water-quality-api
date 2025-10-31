import shap
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import sys
import io
import pickle
from typing import Dict


# This is the fix:
# We are telling Python that the 'RobustSymGAT' class can be found
# in the current module ('app.predictor') even though the pickle file
# thinks it's in '__main__'.
sys.modules['__main__'] = sys.modules[__name__]

# This class definition MUST EXACTLY MATCH the one used for training the model.
# Based on your notebook cell 'COMPLETE SYMGAT WITH FIXED DIVISION BY ZERO ERROR'.

class EnhancedWaterQualityOntology:
    """
    This is the ontology class used within your SymGAT model.
    It must be available to unpickle the model.
    """
    def __init__(self):
        self.standards = {
            'ph': (6.5, 8.5), 'ammonia': (0.001, 0.5), 'bod': (0.001, 5.0),
            'dissolved_oxygen': (6.0, float('inf')), 'nitrate': (0.001, 10.0),
            'temperature': (0.001, 30), 'orthophosphate': (0.001, 0.1),
            'nitrogen': (0.001, 10.0)
        }
        self.pollutant_signatures = {
            'bacterial': {'ammonia': 1.2, 'bod': 1.5, 'dissolved_oxygen': -0.8},
            'chemical': {'ph': 1.5, 'nitrogen': 1.2, 'nitrate': 1.0},
            'organic': {'bod': 2.0, 'dissolved_oxygen': -1.5, 'ammonia': 0.8},
            'agricultural': {'nitrate': 2.0, 'orthophosphate': 1.8, 'nitrogen': 1.5}
        }
    
    def compute_symbolic_features(self, features_dict: Dict[str, float]):
        symbolic_features = []
        param_order = ['ph', 'ammonia', 'bod', 'dissolved_oxygen', 'nitrate']
        for param in param_order:
            if param in features_dict:
                value = features_dict[param]
                min_val, max_val = self.standards.get(param, (0.001, 10))
                penalty = 0.0
                if value < min_val:
                    denominator = min_val if min_val != 0 else 0.001
                    penalty = (min_val - value) / denominator
                elif value > max_val and max_val != float('inf'):
                    denominator = max_val if max_val != 0 else 0.001
                    penalty = (value - max_val) / denominator
                symbolic_features.append(penalty)
            else:
                symbolic_features.append(0.0)
        
        pollutant_order = ['bacterial', 'chemical', 'organic', 'agricultural']
        for pollutant in pollutant_order:
            score, count = 0.0, 0
            for param, weight in self.pollutant_signatures[pollutant].items():
                if param in features_dict and features_dict[param] is not None:
                    std_max = self.standards.get(param, (0.001, 10))[1]
                    if std_max == float('inf'): std_max = 10.0
                    normalized_val = features_dict[param] / (std_max + 1e-8)
                    score += weight * normalized_val
                    count += 1
            symbolic_features.append(score / max(count, 1))
        
        total_penalty = sum(symbolic_features[:5])
        symbolic_features.append(1.0 / (1.0 + total_penalty + 1e-8))
        
        return torch.tensor(symbolic_features, dtype=torch.float32)

class RobustSymGAT(nn.Module):
    """
    Robust SymGAT with improved error handling.
    This definition must match the saved model's architecture.
    """
    def __init__(self, input_dim, symbolic_dim=10, hidden_dim=64, output_dim=1, 
                 num_layers=3, dropout=0.2):
        super().__init__()
        self.symbolic_processor = nn.Sequential(
            nn.Linear(symbolic_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.layers = nn.ModuleList()
        self.input_proj = nn.Linear(input_dim + hidden_dim // 2, hidden_dim)
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)
        )
        self.ontology = EnhancedWaterQualityOntology()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x):
        batch_size = x.shape[0]
        symbolic_list = []
        for i in range(batch_size):
            features_dict = {
                'ammonia': float(x[i, 0]), 'bod': float(x[i, 1]),
                'dissolved_oxygen': float(x[i, 2]), 'ph': float(x[i, 4]),
                'nitrate': float(x[i, 7])
            }
            symbolic_feat = self.ontology.compute_symbolic_features(features_dict)
            symbolic_list.append(symbolic_feat)
        
        symbolic_features = torch.stack(symbolic_list).to(x.device)
        symbolic_encoded = self.symbolic_processor(symbolic_features)
        x_combined = torch.cat([x, symbolic_encoded], dim=1)
        x = self.activation(self.input_proj(x_combined))
        
        for layer in self.layers:
            x_residual = self.activation(layer(x))
            x = x + self.dropout(x_residual)
        
        predictions = self.regression_head(x)
        pollutant_logits = self.classification_head(x)
        
        return {
            'predictions': predictions,
            'pollutant_logits': pollutant_logits,
        }

class Predictor:
    """
    A wrapper class to load the model and make predictions.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # This is a robust way to load a joblib-pickled PyTorch model 
        # that was saved on a GPU.
        self.device = torch.device("cpu")
        
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(model_path, 'rb') as f:
            self.model = CPU_Unpickler(f).load()

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path} and mapped to {self.device} using custom unpickler.")

    def predict(self, input_features: np.ndarray) -> dict:
        """
        Makes a prediction using the loaded model.
        """
        if input_features.ndim == 1:
            input_features = np.expand_dims(input_features, axis=0)
        
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        predictions = outputs['predictions'].cpu().numpy().flatten()
        pollutant_logits = outputs['pollutant_logits'].cpu().numpy()
        pollutant_probs = torch.softmax(torch.tensor(pollutant_logits), dim=1).numpy()
        
        return {
            "predictions": predictions.tolist(),
            "pollutant_probabilities": pollutant_probs.tolist()
        }
    def explain(self, input_features: np.ndarray) -> dict:
        """
        Returns SHAP feature importances for the input.
        """
        if input_features.ndim == 1:
            input_features = np.expand_dims(input_features, axis=0)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)

        # Use SHAP DeepExplainer for PyTorch models
        explainer = shap.DeepExplainer(self.model, input_tensor)
        shap_values = explainer.shap_values(input_tensor)

        # Convert SHAP values to a list for JSON serialization
        feature_importances = np.abs(shap_values[0]).mean(axis=0).tolist()

        return {
            "feature_importances": feature_importances
        }

# Load the model on startup
model_path = os.getenv("MODEL_PATH", "models/RobustSymGAT.pkl")
predictor = Predictor(model_path=model_path)
