import pandas as pd
import numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import shap
import joblib
from datetime import datetime
import lightgbm as lgb # Added for LGBMRegressor
# Optional: ngboost, lifelines for survival, torch for seq models

# Define a placeholder for the number of visits within 24 months (e.g., 4 visits if roughly every 6 months)
n_visits_for_24m = 4

class MotorAgent:
    def __init__(self, data_dir: str = "ppmi_data", seed=42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)
        self.model = None
        self.preprocessor = None
        self.baseline_features = []  # demographics + genetics + baseline motor summary + DaTSCAN SBR etc.

    def load_and_preprocess(self, motor_file="updrs_III.csv"):
        # Load core tables
        motor = pd.read_csv(self.data_dir / motor_file)

        # Rename columns to match expected names in the existing code logic
        motor = motor.rename(columns={
            'patient_id': 'PATNO',
            'assessment_date': 'INFODT',
            'updrs_motor_tremor': 'NUPDRS3_TREMOR',
            'updrs_motor_rigidity': 'NUPDRS3_RIGIDITY',
            'updrs_motor_bradykinesia': 'NUPDRS3_BRADY',
            'updrs_motor_postural_instability': 'NUPDRS3_POSTURAL'
        })

        # Create a synthetic NUPDRS3 total score by summing available motor subscores
        # This assumes these subscores make up the total NUPDRS3 for the purpose of this model
        motor['NUPDRS3'] = motor[['NUPDRS3_TREMOR', 'NUPDRS3_RIGIDITY', 'NUPDRS3_BRADY', 'NUPDRS3_POSTURAL']].sum(axis=1)

        # Assuming all patients in the provided motor file are the target cohort (e.g., PD patients)
        # as there are no separate status or demographic files to filter by.
        pd_pats = motor['PATNO'].unique()
        # No filtering needed explicitly here as we're assuming all are relevant.
        # The motor dataframe already contains only these patients.

        # Time features
        motor['INFODT'] = pd.to_datetime(motor['INFODT'])
        motor = motor.sort_values(['PATNO', 'INFODT']) # Sort by PATNO and date for consistent baseline selection
        motor['months_since_bl'] = motor.groupby('PATNO')['INFODT'].transform(lambda x: (x - x.iloc[0]).dt.days / 30.44)

        # Target: future totals (shift per patient)
        motor['updrs3_future_24m'] = motor.groupby('PATNO')['NUPDRS3'].shift(-n_visits_for_24m)  # define horizon

        # Feature engineering (MIT-level: rich summaries)
        # Define baseline: Ensure self.baseline_df always has one unique record per PATNO.
        # Prioritize delta_t == 0 if available, otherwise take the earliest visit.
        
        # Get all records where delta_t == 0
        baseline_delta_t_0_candidates = motor[motor['delta_t'] == 0].copy()
        
        # For those patients who have delta_t == 0, take the first one (due to prior sort)
        # This ensures uniqueness by PATNO for those with delta_t=0
        baseline_from_delta_t_0 = baseline_delta_t_0_candidates.groupby('PATNO').first().reset_index() if not baseline_delta_t_0_candidates.empty else pd.DataFrame()

        # Identify patients who did NOT have any delta_t == 0 record
        patnos_with_delta_t_0 = baseline_from_delta_t_0['PATNO'].unique()
        motor_remaining = motor[~motor['PATNO'].isin(patnos_with_delta_t_0)]

        # For these remaining patients, take their absolute first visit (after sorting by INFODT)
        first_visits_for_remaining_patients = motor_remaining.groupby('PATNO').first().reset_index() if not motor_remaining.empty else pd.DataFrame()

        # Combine the two sets of baselines
        baseline = pd.concat([baseline_from_delta_t_0, first_visits_for_remaining_patients])
        
        # Ensure no duplicates if some patients somehow ended up in both (shouldn't happen with the current logic, but for robustness)
        baseline = baseline.drop_duplicates(subset=['PATNO'], keep='first')
        
        self.df = motor  # full longitudinal
        self.baseline_df = baseline
        print(f"Loaded: {len(pd_pats)} patients (assuming all are target cohort), motor shape {motor.shape}") # Adjusted print statement
        return self

    def _engineer_features(self):
        """Generates baseline features for the model. For this example, we use the baseline NUPDRS3 score.
        In a real scenario, this would be much richer.
        """
        if self.baseline_df is None:
            raise ValueError("baseline_df is not loaded. Run load_and_preprocess first.")

        # Select relevant baseline features. Here, just NUPDRS3 from the baseline visit.
        # And ensure 'months_since_bl' is available as it's used in the example.
        # We are also dropping any rows where NUPDRS3_BL is NaN as it's a critical feature.
        X = self.baseline_df[['PATNO', 'NUPDRS3', 'months_since_bl']].copy()
        X = X.rename(columns={'NUPDRS3': 'NUPDRS3_BL'})
        X = X.set_index('PATNO')
        return X.dropna()

    def train_progression_model(self, target_col='updrs3_future_24m', use_lgb=True):
        X = self._engineer_features()

        # Align target to the engineered features. The target comes from the full longitudinal dataframe.
        # We link it back to the baseline patient IDs used in X.
        target_series = self.df.set_index('PATNO').groupby(level=0)[target_col].first() # Get the first valid target per patient
        
        # Ensure target_series is aligned with the index of X
        X_aligned, y_aligned = X.align(target_series, join='inner', axis=0)
        y_aligned = y_aligned.dropna() # Drop any NaN targets
        X_aligned = X_aligned.loc[y_aligned.index] # Keep features only for patients with valid targets

        if X_aligned.empty or y_aligned.empty:
            print("Not enough data after feature engineering and target alignment for training.")
            self.model = None # Ensure model is not set if training fails
            return None

        X_train, X_test, y_train, y_test = train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=self.seed)

        if use_lgb:
            model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=7, num_leaves=64,
                                      subsample=0.8, colsample_bytree=0.8, random_state=self.seed)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(50, verbose=False)]) # Use callbacks for early stopping
        else:
            model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.85,
                                     colsample_bytree=0.8, objective='reg:squarederror', random_state=self.seed)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      early_stopping_rounds=50,
                      verbose=False)

        preds = model.predict(X_test)
        print(f"MAE: {mean_absolute_error(y_test, preds):.2f} | R2: {r2_score(y_test, preds):.3f}")

        self.model = model
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer(X_test)  # store for later explain

        joblib.dump(model, f"motor_agent_{datetime.now().strftime('%Y%m%d')}.pkl")
        return model

    def _shap_explain(self, patient_profile: dict):
        """Placeholder for SHAP explanation logic. Needs to be more sophisticated in a full implementation."""
        return {"explanation": "SHAP explanation for top features would go here.", "top_features": list(patient_profile.keys())}

    def predict_and_decide(self, patient_profile: dict) -> dict:
        """Clever decision engine"""
        if self.model is None:
            raise ValueError("Train first")

        # Convert patient_profile to a DataFrame, ensuring it matches model's feature structure
        input_df = pd.DataFrame([patient_profile])

        # Ensure the input_df has the same columns and order as the training data's features
        # This requires the model to have 'feature_names_in_' or similar attribute if LGBM/XGBoost
        # If not available, we assume the input_df already matches the order.
        if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            expected_features = self.model.feature_names_in_
            # Add missing columns with NaN, reorder columns
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df[expected_features]
        else:
            # Fallback if feature names are not directly available from model
            print("Warning: Model feature names not found. Ensure patient_profile matches training feature order.")

        pred = self.model.predict(input_df)[0]
        risk_tier = "HIGH" if pred > 25 else "MED" if pred > 12 else "LOW"  # example thresholds; calibrate!

        decision = {
            "predicted_updrs3_horizon": round(pred, 1),
            "risk_progression": risk_tier,
            "recommendation": self._generate_recommendation(risk_tier, patient_profile),
            "key_drivers": self._shap_explain(patient_profile) if hasattr(self, 'shap_values') else None
        }
        return decision

    def _generate_recommendation(self, tier, profile):
        if tier == "HIGH":
            return "High rapid motor progression risk (esp. if high bradykinesia/gait involvement). Consider earlier dopaminergic therapy, intensive PT, frequent (q3-6mo) monitoring, fall prevention. Reassess DaTSCAN/genetics."
        # ... similar for MED/LOW, tremor-dominant vs akinetic-rigid
        return "Standard monitoring"

# Usage example
agent = MotorAgent(data_dir="/content/") # Changed data_dir to /content/
agent.load_and_preprocess()
agent.train_progression_model()

# Example of a minimal placeholder patient_profile that matches the _engineer_features output
# This assumes a patient with PATNO 3001, NUPDRS3_BL of 20, and months_since_bl of 0
# You would replace these values with actual baseline features for a new patient.
baseline_features_dict = {'NUPDRS3_BL': 20, 'months_since_bl': 0}

# Try to make a prediction if the model was successfully trained
if agent.model is not None:
    decision = agent.predict_and_decide(baseline_features_dict)
    print(decision)
else:
    print("Model could not be trained with available data. Please check data and feature engineering.")