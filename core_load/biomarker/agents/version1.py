from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

# Removed: sklearn.model_selection, sklearn.metrics, lightgbm, shap, joblib as they are no longer needed for a descriptive agent
from datetime import datetime

# Define a placeholder for the number of visits within 24 months (e.g., 4 visits if roughly every 6 months)
n_visits_for_24m = 4 # Still keep this for consistency if needed elsewhere, though not directly used by this agent now

@dataclass
class AgentPayload:
    """Standardized output structure for all agents"""
    agent_name: str
    timestamp: str
    domain_prediction: float # normalized severity score
    uncertainty_variance: float #sigma 2
    feature_importance: Dict[str, float] # e.g {tremor_score: 0.9}
    clinical_narrative: str # local interpretation
    raw_embedding: Optional[np.ndarray] = field(default=None) # latent vector

    def __repr__(self):
        return f"<{self.agent_name} | Pred: {self.domain_prediction:.2f} +/- {self.uncertainty_variance:.2f}>"

class BiomarkerAgent:
    def __init__(self, data_dir: str = "ppmi_data", seed=42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)
        # Removed: self.model, self.shap_explainer, self.target_col as there's no predictive model or target now

    def load_and_preprocess(self, datscan_file="datscan.csv"): # Removed motor_file
        # Load core biomarker tables
        try:
            datscan = pd.read_csv(self.data_dir / datscan_file)
            # Removed motor = pd.read_csv(self.data_dir / motor_file) # No motor data needed
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file: {e}. Check PPMI download folder names or ensure they are provided.")

        # --- Preprocess DaTSCAN data ---
        # Rename columns to match expected format and user's specification
        datscan = datscan.rename(columns={
            'patient_id': 'PATNO',
            'assessment_date': 'INFODT',
            'datscan_left_putamen': 'PUTAMEN_L',
            'datscan_right_putamen': 'PUTAMEN_R',
            'datscan_left_caudate': 'CAUDATE_L',
            'datscan_right_caudate': 'CAUDATE_R'
        })
        datscan['INFODT'] = pd.to_datetime(datscan['INFODT'])
        datscan = datscan.sort_values(['PATNO', 'INFODT'])
        datscan['months_since_bl'] = datscan.groupby('PATNO')['INFODT'].transform(lambda x: (x - x.iloc[0]).dt.days / 30.44)

        # Baseline DaTSCAN features (earliest visit, or within 6 months)
        baseline_dat = datscan[datscan['months_since_bl'] <= 6].groupby('PATNO').first().reset_index()
        baseline_dat['putamen_mean_sbr'] = (baseline_dat['PUTAMEN_L'] + baseline_dat['PUTAMEN_R']) / 2
        baseline_dat['caudate_mean_sbr'] = (baseline_dat['CAUDATE_L'] + baseline_dat['CAUDATE_R']) / 2
        baseline_dat['striatal_asym'] = abs(baseline_dat['PUTAMEN_L'] - baseline_dat['PUTAMEN_R'])
        baseline_dat['low_dat_risk'] = (baseline_dat['putamen_mean_sbr'] < 2.0).astype(int)  # common cutoff ~2.0-2.5

        # --- Store DaTSCAN features ---
        self.df = baseline_dat[['PATNO', 'putamen_mean_sbr', 'caudate_mean_sbr', 'striatal_asym', 'low_dat_risk']]
        self.df = self.df.fillna(0)  # Fill NaNs with 0 for any missing values

        print(f"Biomarker data ready: {len(self.df)} patients with baseline DaTSCAN features")
        return self

    # Removed train_progression_model as there is no predictive model to train

    # Removed _shap_explain as there is no predictive model for SHAP values

    def predict_and_decide(self, patient_bio_profile: dict) -> AgentPayload:
        # No model to train or predict, directly interpret the profile

        # Use 'low_dat_risk' as the primary indicator for domain_prediction (severity score)
        # 0 = lower risk/normal, 1 = higher risk/abnormal (normalized)
        domain_prediction = float(patient_bio_profile.get('low_dat_risk', 0.0)) # Default to 0 if not present

        risk_tier = self._determine_risk_tier(patient_bio_profile)
        recommendation_text = self._generate_recommendation(risk_tier, patient_bio_profile)

        # Heuristic feature importance based on DaTSCAN interpretation
        feature_importances = {
            'putamen_mean_sbr': 0.7, # High importance as a direct measure
            'low_dat_risk': 0.9, # High importance as a categorical risk indicator
            'striatal_asym': 0.3, # Moderate importance
            'caudate_mean_sbr': 0.2 # Lower importance compared to putamen for motor
        }

        uncertainty_val = 0.0 # Placeholder, as no model means no statistical uncertainty

        return AgentPayload(
            agent_name="BiomarkerAgent",
            timestamp=datetime.now().isoformat(),
            domain_prediction=domain_prediction,
            uncertainty_variance=uncertainty_val,
            feature_importance=feature_importances,
            clinical_narrative=recommendation_text,
            raw_embedding=None
        )

    def _determine_risk_tier(self, profile: dict) -> str:
        """Determines risk tier based on DaTSCAN features."""
        if profile.get('low_dat_risk', 0) == 1:
            return "HIGH" # Low putamen SBR indicates significant dopaminergic deficit
        elif profile.get('striatal_asym', 0) > 0.3 or profile.get('caudate_mean_sbr', 99) < 1.5:
            return "MED" # Moderate asymmetry or lower caudate SBR
        else:
            return "LOW" # Normal DaTSCAN findings

    def _generate_recommendation(self, tier: str, profile: dict) -> str:
        """Generates a clinical narrative based on risk tier and DaTSCAN profile."""
        rec = ""
        if tier == "HIGH":
            rec = "High risk based on DaTSCAN findings, indicating significant dopaminergic deficit (low putamen SBR). Recommend confirmation with clinical symptoms, consideration for dopaminergic therapy, and close monitoring for motor progression."
            if profile.get('striatal_asym', 0) > 0.3: # Add more specific details if relevant
                rec += " Significant striatal asymmetry also noted."
        elif tier == "MED":
            rec = "Moderate risk based on DaTSCAN. May show some dopaminergic deficit or asymmetry. Recommend continued clinical monitoring and reassessment of symptoms."
        else: # LOW
            rec = "DaTSCAN findings appear normal. Suggests intact presynaptic dopaminergic neurons. Clinical symptoms may indicate essential tremor or another condition not primarily affecting the nigrostriatal pathway. Routine monitoring as appropriate for clinical presentation."

        return rec

# Usage
agent = BiomarkerAgent(data_dir="/content/") # Changed data_dir to /content/ for common usage in Colab
agent.load_and_preprocess(datscan_file="datscan.csv") # Only datscan.csv is loaded now

# The agent no longer trains a model, so this block is removed.
# if agent.model is None:
#     agent.train_progression_model()

# Example patient profile for prediction. This dictionary needs to contain the DaTSCAN features.
example_patient_bio_profile = {
    'putamen_mean_sbr': 1.8,
    'caudate_mean_sbr': 2.1,
    'striatal_asym': 0.4,
    'low_dat_risk': 1 # 1 if putamen_mean_sbr < 2.0
}

# Since there's no model, we directly call predict_and_decide (renamed for clarity of function, but kept original for now)
# to interpret the profile and generate the AgentPayload.
# No need for the 'if agent.model is not None' check since there is no model.
decision_payload = agent.predict_and_decide(example_patient_bio_profile)
print(decision_payload)
