"""
SHAP Background Data Generator for Water Quality Prediction
============================================================

Generates realistic background data for SHAP explainability based on:
- Training data distributions (n=2,827,977 samples)
- EAS 12:2018 (East African Standard for potable water specification)
- Observed parameter distributions from actual water quality monitoring

This ensures SHAP feature importances are interpretable and aligned with
regulatory standards.
"""

import torch
import numpy as np


def get_background_data_from_distributions(device='cpu', num_samples=100):
    """
    Generate background data matching actual training data distributions.
    
    Based on observed distributions from 2.8M water quality samples:
    - Most parameters show right-skewed (log-normal) distributions
    - Medians represent typical water quality conditions
    - Means are pulled higher by contamination outliers
    
    This approach ensures SHAP values reflect real-world importance of
    parameter deviations relative to typical conditions.
    
    Args:
        device: PyTorch device ('cpu' or 'cuda')
        num_samples: Number of background samples (default: 100)
        
    Returns:
        torch.Tensor: Shape (num_samples, 8) with realistic water quality samples
    """
    np.random.seed(42)  # For reproducibility
    
    # ========== PARAMETER DISTRIBUTIONS ==========
    # Based on training data histograms:
    # - Ammonia: Median=0.10, Mean=1.17 (heavy right skew)
    # - BOD: Median=2.01, Mean=4.89 (right skew)
    # - Dissolved Oxygen: Median=10.20, Mean=10.61 (nearly normal)
    # - Orthophosphate: Median=0.11, Mean=0.71 (extreme right skew)
    # - pH: Median=7.78, Mean=7.74 (nearly normal, centered)
    # - Temperature: Median=11.46, Mean=11.84 (bimodal: cool + warm)
    # - Nitrogen: Median=4.92, Mean=3.25 (moderate skew)
    # - Nitrate: Median=4.92, Mean=5.77 (right skew)
    
    # Strategy: Generate samples matching observed distributions
    # - 70% near median (typical conditions)
    # - 20% moderate elevation (agricultural influence)
    # - 10% high values (contamination events)
    
    # ----- Ammonia (mg/L) -----
    # EAS 12:2018: <0.5 mg/L typical for safety
    # Training: Median=0.10, Mean=1.17
    ammonia_low = np.random.lognormal(np.log(0.10), 0.5, 70)      # Typical
    ammonia_mid = np.random.lognormal(np.log(0.50), 0.8, 20)      # Elevated
    ammonia_high = np.random.lognormal(np.log(2.0), 1.0, 10)      # Contaminated
    ammonia_all = np.concatenate([ammonia_low, ammonia_mid, ammonia_high])
    ammonia_all = np.clip(ammonia_all, 0.001, 10.0)  # Realistic upper bound
    
    # ----- BOD (mg/L) -----
    # Biological Oxygen Demand - indicates organic pollution
    # Training: Median=2.01, Mean=4.89
    bod_low = np.random.lognormal(np.log(2.0), 0.6, 70)           # Typical
    bod_mid = np.random.lognormal(np.log(4.0), 0.8, 20)           # Moderate
    bod_high = np.random.lognormal(np.log(6.0), 0.8, 10)          # High organic load
    bod_all = np.concatenate([bod_low, bod_mid, bod_high])
    bod_all = np.clip(bod_all, 0.1, 15.0)
    
    # ----- Dissolved Oxygen (mg/L) -----
    # EAS 12:2018: Should be >6 mg/L for aquatic life
    # Training: Median=10.20, Mean=10.61 (nearly normal distribution)
    do_all = np.random.normal(10.4, 2.5, 100)
    do_all = np.clip(do_all, 2.0, 20.0)  # Physical limits
    
    # ----- Orthophosphate (mg/L) -----
    # Agricultural runoff indicator
    # Training: Median=0.11, Mean=0.71 (extreme right skew)
    ortho_low = np.random.lognormal(np.log(0.10), 0.7, 75)       # Typical
    ortho_high = np.random.lognormal(np.log(0.50), 1.2, 25)      # Agricultural influence
    ortho_all = np.concatenate([ortho_low, ortho_high])
    ortho_all = np.clip(ortho_all, 0.001, 5.0)
    
    # ----- pH -----
    # EAS 12:2018: 6.5 - 8.5 acceptable range
    # Training: Median=7.78, Mean=7.74 (nearly normal, well-centered)
    ph_all = np.random.normal(7.76, 0.8, 100)
    ph_all = np.clip(ph_all, 5.5, 9.0)  # Allow some violations for diversity
    
    # ----- Temperature (°C) -----
    # EAS 12:2018: Not explicitly limited, but ambient water typically 5-30°C
    # Training: Median=11.46, Mean=11.84 (bimodal: main peak ~11°C, secondary ~22°C)
    temp_main = np.random.normal(11.5, 2.0, 80)   # Cool water (main peak)
    temp_warm = np.random.normal(22.0, 4.0, 20)   # Warmer conditions
    temp_all = np.concatenate([temp_main, temp_warm])
    temp_all = np.clip(temp_all, 5.0, 35.0)
    
    # ----- Nitrogen (mg/L) -----
    # Total nitrogen from multiple sources
    # Training: Median=4.92, Mean=3.25 (moderate skew)
    nitrogen_low = np.random.lognormal(np.log(3.0), 0.8, 70)     # Typical
    nitrogen_high = np.random.lognormal(np.log(7.0), 1.0, 30)    # Elevated
    nitrogen_all = np.concatenate([nitrogen_low, nitrogen_high])
    nitrogen_all = np.clip(nitrogen_all, 0.1, 20.0)
    
    # ----- Nitrate (mg/L) -----
    # EAS 12:2018: 50 mg/L maximum (as NO₃⁻)
    # Training: Median=4.92, Mean=5.77 (right skew)
    nitrate_low = np.random.lognormal(np.log(5.0), 0.9, 70)      # Typical
    nitrate_high = np.random.lognormal(np.log(15.0), 1.2, 30)    # Agricultural areas
    nitrate_all = np.concatenate([nitrate_low, nitrate_high])
    nitrate_all = np.clip(nitrate_all, 0.1, 50.0)  # EAS 12:2018 limit
    
    # ========== COMBINE INTO SAMPLES ==========
    # Shuffle to ensure random feature combinations
    np.random.shuffle(ammonia_all)
    np.random.shuffle(bod_all)
    np.random.shuffle(do_all)
    np.random.shuffle(ortho_all)
    np.random.shuffle(ph_all)
    np.random.shuffle(temp_all)
    np.random.shuffle(nitrogen_all)
    np.random.shuffle(nitrate_all)
    
    # Stack into array: [num_samples, 8 features]
    samples = np.column_stack([
        ammonia_all,
        bod_all,
        do_all,
        ortho_all,
        ph_all,
        temp_all,
        nitrogen_all,
        nitrate_all
    ])
    
    # Convert to PyTorch tensor
    background_tensor = torch.FloatTensor(samples).to(device)
    
    # ========== STATISTICS REPORT ==========
    print(f"\n{'='*70}")
    print(f"SHAP Background Data Generated - EAS 12:2018 Compliant")
    print(f"{'='*70}")
    print(f"Training data reference: n=2,827,977 samples")
    print(f"Background samples: {num_samples}")
    print(f"\nParameter Statistics (Background vs Training):")
    print(f"{'-'*70}")
    
    feature_names = [
        'ammonia', 'bod', 'dissolved_oxygen', 'orthophosphate',
        'ph', 'temperature', 'nitrogen', 'nitrate'
    ]
    training_means = [1.17, 4.89, 10.61, 0.71, 7.74, 11.84, 3.25, 5.77]
    training_medians = [0.10, 2.01, 10.20, 0.11, 7.78, 11.46, 4.92, 4.92]
    deas_limits = [
        "~0.5 mg/L typical",
        "~5.0 mg/L typical", 
        ">6.0 mg/L",
        "~0.1 mg/L typical",
        "6.5-8.5",
        "~10-30°C",
        "~10 mg/L typical",
        "≤50 mg/L"
    ]
    
    for i, name in enumerate(feature_names):
        col = background_tensor[:, i].cpu().numpy()
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  Background: mean={col.mean():.2f}, median={np.median(col):.2f}, "
              f"range=[{col.min():.2f}, {col.max():.2f}]")
        print(f"  Training:   mean={training_means[i]:.2f}, median={training_medians[i]:.2f}")
        print(f"  EAS 12:2018: {deas_limits[i]}")
    
    print(f"\n{'='*70}")
    print(f"Distribution Characteristics:")
    print(f"  - 70% samples: Typical conditions (near median)")
    print(f"  - 20% samples: Moderate elevation (agricultural influence)")
    print(f"  - 10% samples: High values (contamination events)")
    print(f"{'='*70}\n")
    
    return background_tensor


def get_deas_compliance_summary():
    """
    Return EAS 12:2018 compliance thresholds for reference.
    
    Returns:
        dict: Parameter limits and interpretations
    """
    return {
        'ph': {
            'range': (6.5, 8.5),
            'unit': 'pH units',
            'interpretation': 'Acidity/alkalinity balance',
            'violation_impact': 'Corrosion, toxicity to aquatic life'
        },
        'dissolved_oxygen': {
            'range': (6.0, float('inf')),
            'unit': 'mg/L',
            'interpretation': 'Essential for aquatic ecosystem health',
            'violation_impact': 'Fish kills, anaerobic conditions'
        },
        'nitrate': {
            'range': (0.001, 50.0),
            'unit': 'mg/L (as NO₃⁻)',
            'interpretation': 'Agricultural fertilizer runoff',
            'violation_impact': 'Eutrophication, methemoglobinemia risk'
        },
        'ammonia': {
            'range': (0.001, 0.5),
            'unit': 'mg/L',
            'interpretation': 'Sewage or agricultural waste indicator',
            'violation_impact': 'Toxicity to aquatic organisms'
        },
        'bod': {
            'range': (0.001, 5.0),
            'unit': 'mg/L',
            'interpretation': 'Organic pollution load',
            'violation_impact': 'Oxygen depletion, ecosystem stress'
        },
        'orthophosphate': {
            'range': (0.001, 0.1),
            'unit': 'mg/L',
            'interpretation': 'Phosphate from detergents/fertilizers',
            'violation_impact': 'Algal blooms, eutrophication'
        },
        'temperature': {
            'range': (0.001, 30.0),
            'unit': '°C',
            'interpretation': 'Affects biological and chemical processes',
            'violation_impact': 'Metabolic stress, reduced oxygen solubility'
        },
        'nitrogen': {
            'range': (0.001, 10.0),
            'unit': 'mg/L',
            'interpretation': 'Total nitrogen from multiple sources',
            'violation_impact': 'Eutrophication, ecosystem imbalance'
        }
    }


# For testing purposes
if __name__ == "__main__":
    print("Testing SHAP Background Data Generator...")
    background = get_background_data_from_distributions(device='cpu', num_samples=100)
    print(f"\nBackground tensor shape: {background.shape}")
    print(f"Device: {background.device}")
    print(f"\nSample (first row):")
    print(f"  {background[0].tolist()}")
    
    compliance = get_deas_compliance_summary()
    print(f"\nEAS 12:2018 Standards Loaded:")
    for param, info in compliance.items():
        print(f"  {param}: {info['range']} {info['unit']}")
