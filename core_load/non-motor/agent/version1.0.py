from dataclasses import dataclass, field
from typing import List, Dict, Optional # Added for AgentPayload
import pandas as pd
import numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import shap
import joblib
from datetime import datetime

# Define a placeholder for the number of visits within 24 months (e.g., 4 visits if roughly every 6 months)
n_visits_for_24m = 4 # Consistent with MotorAgent

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

class NonMotorAgent:
    def __init__(self, data_dir: str = "/content/", seed=42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)
        self.model = None
        self.preprocessor = None # Added for consistency with MotorAgent
        self.baseline_features = [] # Added for consistency with MotorAgent
        self.shap_explainer = None
        # key_domains removed as features are explicitly defined by the input data structure

    def load_and_preprocess(self, non_motor_file="merged_non_motor_data.csv"):
        # Load core non-motor table
        try:
            # Read CSV assuming no header, then assign columns manually
            column_names = [
                'patient_id', 'assessment_date_x', 'updrs_nonmotor_sleep',
                'assessment_date_y', 'updrs_nonmotor_depression',
                'assessment_date', 'updrs_nonmotor_cognitive'
            ]
            non_motor_df = pd.read_csv(self.data_dir / non_motor_file, header=None, names=column_names)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file: {e}. Ensure '{non_motor_file}' is available.")

        # Debug print: Show columns immediately after loading and assigning names
        print("Columns after initial CSV load and name assignment:", non_motor_df.columns.tolist())

        # Rename 'patient_id' to 'PATNO'
        non_motor_df = non_motor_df.rename(columns={'patient_id': 'PATNO'})

        # Explicitly convert score columns to numeric, coercing errors
        score_cols = ['updrs_nonmotor_sleep', 'updrs_nonmotor_depression', 'updrs_nonmotor_cognitive']
        for col in score_cols:
            if col in non_motor_df.columns:
                non_motor_df[col] = pd.to_numeric(non_motor_df[col], errors='coerce')

        # Combine multiple assessment date columns into a single 'INFODT'
        # Prioritize 'assessment_date', then 'assessment_date_x', then 'assessment_date_y'
        if 'assessment_date' in non_motor_df.columns:
            non_motor_df['INFODT'] = pd.to_datetime(non_motor_df['assessment_date'], errors='coerce')
        else:
            non_motor_df['INFODT'] = pd.NaT # Initialize with Not a Time if not present

        if 'assessment_date_x' in non_motor_df.columns:
            non_motor_df['INFODT'] = non_motor_df['INFODT'].fillna(
                pd.to_datetime(non_motor_df['assessment_date_x'], errors='coerce')
            )
        if 'assessment_date_y' in non_motor_df.columns:
            non_motor_df['INFODT'] = non_motor_df['INFODT'].fillna(
                pd.to_datetime(non_motor_df['assessment_date_y'], errors='coerce')
            )
            
        # Drop the original date columns to avoid redundancy and potential confusion
        original_date_cols = [col for col in ['assessment_date_x', 'assessment_date_y', 'assessment_date'] if col in non_motor_df.columns]
        non_motor_df = non_motor_df.drop(columns=original_date_cols)

        # Drop rows where the combined primary date is still missing
        non_motor_df = non_motor_df.dropna(subset=['INFODT']).copy()

        # Convert primary assessment date to datetime (already done, but ensure final type)
        non_motor_df['INFODT'] = pd.to_datetime(non_motor_df['INFODT'], errors='coerce')
        non_motor_df = non_motor_df.sort_values(['PATNO', 'INFODT'])

        # Print columns for debugging
        print("Columns after initial processing:", non_motor_df.columns.tolist())

        # Calculate months since baseline for each patient based on primary INFODT
        non_motor_df['months_since_bl'] = non_motor_df.groupby('PATNO')['INFODT'].transform(lambda x: (x - x.iloc[0]).dt.days / 30.44)

        # Define targets for future progression (e.g., 24 months = n_visits_for_24m visits)
        non_motor_df['updrs_nonmotor_sleep_future_24m'] = non_motor_df.groupby('PATNO')['updrs_nonmotor_sleep'].shift(-n_visits_for_24m)
        non_motor_df['updrs_nonmotor_depression_future_24m'] = non_motor_df.groupby('PATNO')['updrs_nonmotor_depression'].shift(-n_visits_for_24m)
        non_motor_df['updrs_nonmotor_cognitive_future_24m'] = non_motor_df.groupby('PATNO')['updrs_nonmotor_cognitive'].shift(-n_visits_for_24m)

        # Baseline definition: Similar to MotorAgent, ensure a unique baseline record per PATNO.
        # Taking the first record after sorting by INFODT as baseline.
        baseline = non_motor_df.groupby('PATNO').first().reset_index()

        self.df = non_motor_df  # full longitudinal
        self.baseline_df = baseline
        print(f"Loaded {len(non_motor_df['PATNO'].unique())} patients | Non-motor data shape {non_motor_df.shape}")
        return self

    def _engineer_features(self):
        """Generates baseline features for the model."""
        if self.baseline_df is None:
            raise ValueError("baseline_df is not loaded. Run load_and_preprocess first.")

        # Select relevant baseline features.
        feature_cols = ['PATNO', 'updrs_nonmotor_sleep', 'updrs_nonmotor_depression', 'updrs_nonmotor_cognitive', 'months_since_bl']
        X = self.baseline_df[feature_cols].copy()
        X = X.rename(columns={
            'updrs_nonmotor_sleep': 'updrs_nonmotor_sleep_BL',
            'updrs_nonmotor_depression': 'updrs_nonmotor_depression_BL',
            'updrs_nonmotor_cognitive': 'updrs_nonmotor_cognitive_BL'
        })
        X = X.set_index('PATNO')
        return X.dropna()

    def train_progression_model(self, target_col='updrs_nonmotor_sleep_future_24m', domain='sleep'):
        X = self._engineer_features()

        # Align target to the engineered features.
        target_series = self.df.set_index('PATNO').groupby(level=0)[target_col].first()

        # Ensure target_series is aligned with the index of X
        X_aligned, y_aligned = X.align(target_series, join='inner', axis=0)
        y_aligned = y_aligned.dropna() # Drop any NaN targets
        X_aligned = X_aligned.loc[y_aligned.index] # Keep features only for patients with valid targets

        if X_aligned.empty or y_aligned.empty:
            print(f"Not enough data after feature engineering and target alignment for training '{target_col}'.")
            self.model = None
            return None

        X_train, X_test, y_train, y_test = train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=self.seed)

        model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=7, num_leaves=64,
                                  subsample=0.8, colsample_bytree=0.8, random_state=self.seed)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        preds = model.predict(X_test)
        print(f"{domain} {target_col} - MAE: {mean_absolute_error(y_test, preds):.2f} | R2: {r2_score(y_test, preds):.3f}")

        self.model = model
        self.shap_explainer = shap.TreeExplainer(model)
        self.shap_values = self.shap_explainer(X_test)

        joblib.dump(model, f"nonmotor_agent_{domain}_{datetime.now().strftime('%Y%m%d')}.pkl")
        return model

    def _shap_explain(self, patient_profile: dict, input_df: pd.DataFrame) -> Dict[str, float]:
        """Generates SHAP explanation for a single patient profile and returns feature importances."""
        feature_importances = {} 
        if self.model is None:
            print("Warning: Model not trained, cannot generate SHAP explanation.")
            return {}

        try:
            explainer = shap.TreeExplainer(self.model)
            # Calculate SHAP values for the specific input_df
            shap_values_instance = explainer(input_df)

            # Extract feature names (from the model's training features) and SHAP values
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                feature_names = self.model.feature_names_in_
            else:
                # Fallback if model doesn't expose feature_names_in_
                feature_names = list(input_df.columns) # Assume input_df columns are feature names
                print("Warning: Model feature names not found. Using input_df columns for SHAP explanation.")

            # For a single instance, shap_values_instance.values will be a 1D array
            shap_values_single = shap_values_instance.values[0] if len(shap_values_instance.values.shape) > 1 else shap_values_instance.values

            # Create a dictionary of feature importances (absolute SHAP values for simplicity)
            for i, feature in enumerate(feature_names):
                if i < len(shap_values_single):
                    feature_importances[feature] = float(abs(shap_values_single[i])) # Ensure float type

        except Exception as e:
            print(f"Error generating SHAP explanation for patient: {e}")

        return feature_importances

    def predict_and_decide(self, patient_profile: dict, domain: str = 'sleep') -> AgentPayload:
        if self.model is None:
            raise ValueError("Train model first")

        input_df = pd.DataFrame([patient_profile])

        # Ensure the input_df has the same columns and order as the training data's features
        if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            expected_features = self.model.feature_names_in_
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = np.nan # Add missing columns with NaN
            input_df = input_df[expected_features]
        else:
            print("Warning: Model feature names not found. Ensure patient_profile matches training feature order.")
            # Fallback: try to reorder based on assumed baseline features
            assumed_baseline_features = ['updrs_nonmotor_sleep_BL', 'updrs_nonmotor_depression_BL', 'updrs_nonmotor_cognitive_BL', 'months_since_bl']
            input_df = input_df.reindex(columns=assumed_baseline_features, fill_value=np.nan)


        pred = self.model.predict(input_df)[0]

        risk_tier = self._determine_risk_tier(pred, domain)
        recommendation_text = self._generate_recommendation(risk_tier, domain, patient_profile)
        feature_importances = self._shap_explain(patient_profile, input_df)

        uncertainty_val = 0.0 # Placeholder

        return AgentPayload(
            agent_name=f"NonMotorAgent_{domain}",
            timestamp=datetime.now().isoformat(),
            domain_prediction=float(pred),
            uncertainty_variance=uncertainty_val,
            feature_importance=feature_importances,
            clinical_narrative=recommendation_text,
            raw_embedding=None
        )

    def _determine_risk_tier(self, prediction: float, domain: str) -> str:
        """Determines risk tier based on prediction and domain-specific thresholds."""
        if domain == 'sleep':
            # Example thresholds for updrs_nonmotor_sleep (hypothetical)
            # Higher score indicates worse sleep
            return "HIGH" if prediction > 10 else "MED" if prediction > 5 else "LOW"
        elif domain == 'depression':
            # Example thresholds for updrs_nonmotor_depression (hypothetical)
            # Higher score indicates worse depression
            return "HIGH" if prediction > 15 else "MED" if prediction > 8 else "LOW"
        elif domain == 'cognitive':
            # Example thresholds for updrs_nonmotor_cognitive (hypothetical)
            # Lower score indicates worse cognition
            return "HIGH" if prediction < 20 else "MED" if prediction < 25 else "LOW"
        else:
            return "UNKNOWN"

    def _generate_recommendation(self, tier: str, domain: str, profile: dict) -> str:
        """Generates a clinical narrative based on risk tier and domain."""
        if domain == 'sleep':
            if tier == "HIGH":
                return "High risk of significant sleep disturbance: Consider polysomnography, evaluate for REM sleep behavior disorder (RBD), restless legs syndrome. Recommend sleep hygiene education, potential pharmacotherapy."
            elif tier == "MED":
                return "Moderate sleep concerns: Advise on sleep hygiene. Monitor for worsening symptoms. Consider sleep diary or actigraphy."
            else:
                return "Low sleep disturbance risk: Continue monitoring. Reinforce healthy sleep habits."
        elif domain == 'depression':
            if tier == "HIGH":
                return "High risk of depression: Initiate formal psychiatric evaluation, consider antidepressant therapy, psychotherapy. Screen for suicidality."
            elif tier == "MED":
                return "Moderate depressive symptoms: Monitor mood, encourage social engagement and exercise. Consider referral for counseling."
            else:
                return "Low depressive symptom risk: Routine psychological well-being check."
        elif domain == 'cognitive':
            if tier == "HIGH":
                return "High risk of cognitive impairment: Recommend comprehensive neuropsychological assessment. Evaluate for dementia, consider cognitive enhancing medications (if appropriate). Cognitive rehabilitation."
            elif tier == "MED":
                return "Moderate cognitive concerns: Encourage cognitive stimulation, physical activity, healthy diet. Regular cognitive screening recommended."
            else:
                return "Low cognitive risk: Continue healthy lifestyle, routine cognitive monitoring."
        else:
            return "Standard monitoring for non-motor symptoms."


# Example Usage (modified to reflect the new agent)
agent = NonMotorAgent(data_dir="/content/") # Ensure this path is correct for your merged_non_motor_data.csv
agent.load_and_preprocess(non_motor_file="merged_non_motor_data.csv") # Specify the merged file

# Train for sleep progression
# Check if the model is not none, to avoid training multiple times if cell is rerun.
if agent.model is None:
    agent.train_progression_model(target_col='updrs_nonmotor_sleep_future_24m', domain='sleep')

# Example patient profile for prediction (replace with actual baseline features)
# This dict needs to match the '_engineer_features' output for feature names
# E.g., if _engineer_features creates 'updrs_nonmotor_sleep_BL', 'updrs_nonmotor_depression_BL', 'updrs_nonmotor_cognitive_BL', 'months_since_bl'
baseline_patient_dict = {
    'updrs_nonmotor_sleep_BL': 7.0, # Example baseline score for sleep
    'updrs_nonmotor_depression_BL': 10.0, # Example baseline score for depression
    'updrs_nonmotor_cognitive_BL': 22.0, # Example baseline score for cognitive
    'months_since_bl': 0.0
}

# Try to make a prediction if the model was successfully trained
if agent.model is not None:
    # Make a prediction for sleep domain
    decision_payload_sleep = agent.predict_and_decide(baseline_patient_dict, domain='sleep')
    print("Sleep Agent Decision:", decision_payload_sleep)

    # To train for another domain, you would typically create another instance of the agent
    # or manage multiple models within one agent (e.g., a dictionary of models).
    # For this example, we demonstrate one domain. If you want to run for depression:
    # agent_depression = NonMotorAgent(data_dir="/content/")
    # agent_depression.load_and_preprocess(non_motor_file="merged_non_motor_data.csv")
    # agent_depression.train_progression_model(target_col='updrs_nonmotor_depression_future_24m', domain='depression')
    # decision_payload_depression = agent_depression.predict_and_decide(baseline_patient_dict, domain='depression')
    # print("Depression Agent Decision:", decision_payload_depression)
else:
    print("NonMotorAgent model could not be trained with available data for 'sleep' domain. Please check data and feature engineering.")
