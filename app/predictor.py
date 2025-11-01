import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
import sys
import os
import io
import pickle
from typing import Dict
warnings.filterwarnings('ignore')

# ====================================================================
# 1. FIXED SYMGAT LAYER (The component that caused the original errors)
# ====================================================================

class SymGATLayer(nn.Module):
    """
    Symmetric Graph Attention Layer (SymGAT)
    FIXED: 
    1. Uses torch.matmul for attention calculation (Fixes 'Parameter' object not callable).
    2. Uses .item() for fill_diagonal_ (Fixes original TypeError).
    """
    def __init__(self, in_features, out_features, dropout, negative_slope, concat=True):
        super(SymGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.concat = concat
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism parameter 'a'
        self.att = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.att.data, gain=1.414)

        self.activation = nn.ELU()
        
    def forward(self, h, edge_index):
        # 1. Linear transformation
        h_trans = self.linear(h) 

        # 2. Compute attention scores (e_ij = a(Wh_i || Wh_j))
        row, col = edge_index
        h_row = h_trans[row]
        h_col = h_trans[col]
        
        e = torch.cat([h_row, h_col], dim=1) # (num_edges, 2 * out_features)
        
        # ✅ FINAL FIX: Use torch.matmul to apply the attention vector 'a'
        edge_e = torch.matmul(e, self.att) # (num_edges, 1)
        
        edge_e = F.leaky_relu(edge_e, self.negative_slope)

        # 3. Aggregate attention scores into an attention matrix
        num_nodes = h.size(0)
        attention = torch.zeros((num_nodes, num_nodes), device=h.device)
        attention[row, col] = edge_e.squeeze()

        # 4. Apply the Symmetric Self-Attention Fix
        attention = (attention + attention.T) / 2.0

        # 5. Add self-loops with high attention
        fill_value = 1.0
        if len(edge_e) > 0:
            # FIX from original issue: Use .item()
            fill_value = edge_e.max().item()
            
        attention.fill_diagonal_(fill_value)

        # 6. Softmax to get attention coefficients (alpha_ij)
        attention = attention.softmax(dim=1)

        # 7. Apply attention to neighbors' features
        h_prime = torch.matmul(attention, h_trans)

        # 8. Apply activation and dropout
        if self.concat:
            h_prime = self.activation(h_prime)
        
        return F.dropout(h_prime, p=self.dropout, training=self.training), attention


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
        # Symbolic feature processor
        self.symbolic_processor = nn.Sequential(
            nn.Linear(symbolic_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        # Initial projection and SymGAT layers
        self.layers = nn.ModuleList()
        input_to_gat = input_dim + hidden_dim // 2
        # Initial SymGAT Layer (first layer handles combined features)
        self.layers.append(SymGATLayer(
            input_to_gat, hidden_dim, dropout, negative_slope=0.2, concat=True
        ))
        # Subsequent SymGAT Layers
        for i in range(1, num_layers):
            self.layers.append(SymGATLayer(
                hidden_dim, hidden_dim, dropout, negative_slope=0.2, concat=True
            ))
        # Output heads
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
            nn.Linear(hidden_dim // 2, 4)  # 4 pollutant types
        )
        self.ontology = EnhancedWaterQualityOntology()
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x, edge_index, seed_indices):
        batch_size = x.shape[0]
        # 1. Symbolic Feature Generation
        symbolic_list = []
        for i in range(batch_size):
            try:
                features_dict = {
                    'ammonia': float(x[i, 0]),
                    'bod': float(x[i, 1]) if x.shape[1] > 1 else 2.0,
                    'dissolved_oxygen': float(x[i, 2]) if x.shape[1] > 2 else 8.0,
                    'ph': float(x[i, 4]) if x.shape[1] > 4 else 7.0,
                    'nitrate': float(x[i, 7]) if x.shape[1] > 7 else 5.0
                }
                symbolic_feat = self.ontology.compute_symbolic_features(features_dict)
                symbolic_list.append(symbolic_feat)
            except Exception as e:
                default_features = torch.tensor([0.0] * 10, dtype=torch.float32)
                symbolic_list.append(default_features)
        symbolic_features = torch.stack(symbolic_list).to(x.device)
        symbolic_encoded = self.symbolic_processor(symbolic_features)
        # 2. Combine features
        x = torch.cat([x, symbolic_encoded], dim=1)
        # 3. Apply SymGAT layers
        for i, layer in enumerate(self.layers):
            h_in = x if i == 0 else h_out
            h_out, _ = layer(h_in, edge_index)
            # Apply residual connection (only works if in_dim == out_dim)
            if i > 0 and h_in.shape[1] == h_out.shape[1]:
                h_out = h_in + h_out
            x = h_out
        final_embedding = x 
        predictions = self.regression_head(final_embedding)
        pollutant_logits = self.classification_head(final_embedding)
        return {
            'predictions': predictions,
            'pollutant_logits': pollutant_logits,
            'embeddings': final_embedding
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
        # DEBUG: log inputs for traceability (will appear in server logs)
        try:
            print(f"[Predictor] input_features (ndarray) shape={input_features.shape} values=\n{input_features}")
        except Exception:
            pass
        batch_size = input_tensor.shape[0]
        # Create a fully connected dummy edge_index for batch_size nodes
        row = torch.arange(batch_size).repeat_interleave(batch_size)
        col = torch.arange(batch_size).repeat(batch_size)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0).to(self.device)
        seed_indices = torch.arange(batch_size).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor, edge_index, seed_indices)
        # Raw regression outputs from the model (may be unbounded)
        raw_predictions = outputs['predictions']
        # Convert to probability-like score in [0,1] using sigmoid for thresholding
        try:
            sigmoid_preds = torch.sigmoid(raw_predictions).cpu().numpy().flatten()
        except Exception:
            # Fallback in case raw_predictions is already numpy
            sigmoid_preds = (1.0 / (1.0 + np.exp(-raw_predictions.cpu().numpy().flatten())))
        predictions = sigmoid_preds
        pollutant_logits = outputs['pollutant_logits'].cpu().numpy()
        pollutant_probs = torch.softmax(torch.tensor(pollutant_logits), dim=1).numpy()
        # DEBUG: log raw and sigmoid model outputs
        try:
            print(f"[Predictor] raw_predictions=\n{raw_predictions.detach().cpu().numpy().flatten()}")
            print(f"[Predictor] sigmoid_predictions=\n{predictions}")
            print(f"[Predictor] pollutant_logits=\n{pollutant_logits}")
            print(f"[Predictor] pollutant_probabilities=\n{pollutant_probs}")
        except Exception:
            pass
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
        batch_size = input_tensor.shape[0]
        # Create a fully connected dummy edge_index for batch_size nodes
        row = torch.arange(batch_size).repeat_interleave(batch_size)
        col = torch.arange(batch_size).repeat(batch_size)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0).to(self.device)
        seed_indices = torch.arange(batch_size).to(self.device)

        # Use SHAP Explainer for PyTorch models
        def model_with_graph(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            preds = self.model(x_tensor, edge_index, seed_indices)['predictions']
            arr = preds.detach().cpu().numpy()
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr

        # Use a small background for SHAP
        background = input_features if input_features.shape[0] < 100 else input_features[:100]
        try:
            explainer = shap.Explainer(model_with_graph, background)
            shap_values = explainer(input_features)
            feature_importances = np.abs(shap_values.values).mean(axis=0).tolist()
        except Exception:
            explainer = shap.KernelExplainer(model_with_graph, background)
            shap_values = explainer.shap_values(input_features)
            feature_importances = np.abs(shap_values).mean(axis=0).tolist()
        # DEBUG: log SHAP importances
        try:
            print(f"[Predictor] feature_importances=\n{feature_importances}")
        except Exception:
            pass
        return {
            "feature_importances": feature_importances
        }

# Load the model on startup
model_path = os.getenv("MODEL_PATH", "models/RobustSymGAT.pkl")
predictor = Predictor(model_path=model_path)
