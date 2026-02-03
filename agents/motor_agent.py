"""
Unified Motor Agent for Parkinson's Disease Progression

Consolidates motor symptom analysis from UPDRS-III (Unified Parkinson's Disease Rating Scale - Part III)
Merges functionality from:
- agents/motor.py
- core_load/motor agent/agents/version1/version1.0.py

Features:
- Tremor, rigidity, bradykinesia, postural instability assessment
- 24-month progression forecasting
- LightGBM/XGBoost models
- SHAP-based interpretability
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

from core.base_agent import ClinicalAgent
from core.protocol import AgentPayload, ModelMetadata


class MotorAgent(ClinicalAgent):
    """Specialized agent for motor symptom assessment and progression prediction"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        model_path: Optional[str] = None,
        config: Optional[Any] = None
    ):
        super().__init__(
            agent_name="MotorAgent",
            data_path=data_path,
            model_path=model_path,
            config=config
        )
        
        # Motor-specific configuration
        self.max_updrs3_score = 132.0 if not config else config.motor.max_updrs3_score
        self.n_visits_24m = 4 if not config else config.data.n_visits_for_24m
        
    def ingest_ppmi_data(self) -> pd.DataFrame:
        """Load UPDRS-III motor assessment data
        
        Supports multiple file formats:
        1. Standard PPMI MDS_UPDRS_Part_III.csv
        2. Preprocessed formatted_parkinsons_dataset_dataset1.csv
        
        Returns:
            DataFrame with motor assessment data
        """
        df = pd.read_csv(self.data_path)
        
        # Detect format and standardize column names
        if 'patient_id' in df.columns:
            # Format: formatted dataset
            df = df.rename(columns={
                'patient_id': 'PATNO',
                'assessment_date': 'INFODT',
                'updrs_motor_tremor': 'NUPDRS3_TREMOR',
                'updrs_motor_rigidity': 'NUPDRS3_RIGIDITY',
                'updrs_motor_bradykinesia': 'NUPDRS3_BRADY',
                'updrs_motor_postural_instability': 'NUPDRS3_POSTURAL'
            })
        
        # Ensure required columns exist
        required_cols = ['PATNO', 'INFODT']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        return df
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess motor data
        
        Steps:
        1. Parse dates and sort temporally
        2. Calculate total UPDRS-III score from subscores
        3. Calculate months since baseline
        4. Create future targets for progression prediction
        5. Define baseline records
        
        Args:
            data: Raw motor data
            
        Returns:
            Preprocessed DataFrame
        """
        data = data.copy()
        
        # Parse dates
        data['INFODT'] = pd.to_datetime(data['INFODT'], errors='coerce')
        data = data.dropna(subset=['INFODT'])
        data = data.sort_values(['PATNO', 'INFODT'])
        
        # Calculate total UPDRS-III score
        # Check if subscores exist, otherwise look for total
        subscore_cols = ['NUPDRS3_TREMOR', 'NUPDRS3_RIGIDITY', 'NUPDRS3_BRADY', 'NUPDRS3_POSTURAL']
        if all(col in data.columns for col in subscore_cols):
            data['NUPDRS3'] = data[subscore_cols].sum(axis=1)
        elif 'NUPDRS3' not in data.columns:
            # Try to find NP3 columns (PPMI standard naming)
            np3_cols = [c for c in data.columns if 'NP3' in c]
            if np3_cols:
                data['NUPDRS3'] = data[np3_cols].sum(axis=1)
            else:
                raise ValueError("Cannot calculate NUPDRS3 score - missing subscore columns")
        
        # Calculate months since baseline
        data['months_since_bl'] = data.groupby('PATNO')['INFODT'].transform(
            lambda x: (x - x.iloc[0]).dt.days / 30.44
        )
        
        # Create delta_t for baseline identification (if not exists)
        if 'delta_t' not in data.columns:
            data['delta_t'] = data['months_since_bl'].round().astype(int)
        
        # Create future target (24-month horizon)
        data['updrs3_future_24m'] = data.groupby('PATNO')['NUPDRS3'].shift(-self.n_visits_24m)
        
        # Define baseline (prioritize delta_t==0, otherwise first visit)
        baseline_delta_0 = data[data['delta_t'] == 0].groupby('PATNO').first().reset_index()
        patnos_with_delta_0 = baseline_delta_0['PATNO'].unique()
        
        remaining = data[~data['PATNO'].isin(patnos_with_delta_0)]
        baseline_first_visit = remaining.groupby('PATNO').first().reset_index()
        
        self.baseline_df = pd.concat([baseline_delta_0, baseline_first_visit])
        self.baseline_df = self.baseline_df.drop_duplicates(subset=['PATNO'], keep='first')
        
        return data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer baseline features for progression prediction
        
        Features:
        - Baseline UPDRS-III total score
        - Baseline motor subscores (tremor, rigidity, bradykinesia, postural)
        - Months since baseline (for temporal modeling)
        
        Returns:
            DataFrame with engineered features (indexed by PATNO)
        """
        if self.baseline_df is None:
            raise ValueError("baseline_df not loaded. Run load_and_prepare() first")
        
        # Select baseline features
        feature_cols = ['PATNO', 'NUPDRS3', 'months_since_bl']
        
        # Add subscores if available
        subscore_cols = ['NUPDRS3_TREMOR', 'NUPDRS3_RIGIDITY', 'NUPDRS3_BRADY', 'NUPDRS3_POSTURAL']
        available_subscores = [col for col in subscore_cols if col in self.baseline_df.columns]
        feature_cols.extend(available_subscores)
        
        X = self.baseline_df[feature_cols].copy()
        
        # Rename UPDRS3 to indicate baseline
        X = X.rename(columns={'NUPDRS3': 'NUPDRS3_BL'})
        X = X.set_index('PATNO')
        
        return X.dropna()
    
    def analyze(self, patient_id: Optional[str] = None, patient_profile: Optional[Dict] = None) -> AgentPayload:
        """Analyze motor symptoms for a patient
        
        Two modes:
        1. With patient_id: Load from self.df
        2. With patient_profile: Use provided features
        
        Args:
            patient_id: Patient identifier
            patient_profile: Dictionary of baseline features
            
        Returns:
            AgentPayload with motor assessment
        """
        if patient_profile is None and patient_id is None:
            raise ValueError("Must provide either patient_id or patient_profile")
        
        # Mode 1: Load from data
        if patient_id and self.df is not None:
            patient_data = self.df[self.df['PATNO'] == int(patient_id)]
            if patient_data.empty:
                return self._generate_empty_payload(patient_id)
            
            latest = patient_data.iloc[-1]
            latest_score = latest['NUPDRS3']
            normalized_severity = min(latest_score / self.max_updrs3_score, 1.0)
            
            # Uncertainty: lower if ON medication state, higher if OFF
            uncertainty = 0.1 if 'ON' in str(latest.get('PAG_NAME', '')) else 0.3
            
            # Feature importance (simple heuristic if no model)
            feature_importance = {
                "tremor": 0.6,
                "rigidity": 0.4,
                "bradykinesia": 0.5,
                "postural_instability": 0.3
            }
            
            narrative = f"Patient shows motor score of {latest_score:.0f}/{self.max_updrs3_score:.0f}. "
            if normalized_severity > 0.5:
                narrative += "Moderate to severe motor impairment."
            elif normalized_severity > 0.25:
                narrative += "Mild to moderate motor symptoms."
            else:
                narrative += "Minimal motor impairment."
            
            return AgentPayload(
                agent_name=self.agent_name,
                timestamp=str(latest['INFODT']),
                domain_prediction=normalized_severity,
                uncertainty_variance=uncertainty,
                feature_importance=feature_importance,
                clinical_narrative=narrative,
                patient_id=patient_id,
                model_metadata=self.model_metadata
            )
        
        # Mode 2: Use model prediction
        if patient_profile and self.model:
            prediction, feature_importance = self.predict(patient_profile)
            
            # Normalize prediction if needed
            if prediction > self.max_updrs3_score:
                prediction = self.max_updrs3_score
            normalized_prediction = prediction / self.max_updrs3_score
            
            # Determine risk tier
            risk_tier = self._determine_risk_tier(prediction)
            narrative = self._generate_recommendation(risk_tier, patient_profile, prediction)
            
            # Estimate uncertainty (could be improved with model uncertainty quantification)
            uncertainty = 0.15  # Default
            
            payload = AgentPayload(
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
                domain_prediction=normalized_prediction,
                uncertainty_variance=uncertainty,
                feature_importance=feature_importance,
                clinical_narrative=narrative,
                model_metadata=self.model_metadata
            )
            payload.calculate_confidence_interval()
            
            return payload
        
        # Fallback
        return self._generate_empty_payload(patient_id)
    
    def _determine_risk_tier(self, predicted_score: float) -> str:
        """Determine risk tier based on predicted UPDRS-III score
        
        Thresholds (configurable):
        - HIGH: > 25 (severe motor progression)
        - MED: 12-25 (moderate progression)
        - LOW: < 12 (mild progression)
        
        Args:
            predicted_score: Predicted UPDRS-III score
            
        Returns:
            Risk tier string
        """
        high_threshold = 25.0 if not self.config else self.config.motor.high_risk_threshold
        med_threshold = 12.0 if not self.config else self.config.motor.med_risk_threshold
        
        if predicted_score > high_threshold:
            return "HIGH"
        elif predicted_score > med_threshold:
            return "MED"
        else:
            return "LOW"
    
    def _generate_recommendation(self, risk_tier: str, profile: Dict, predicted_score: float) -> str:
        """Generate clinical recommendation based on risk tier
        
        Args:
            risk_tier: HIGH, MED, or LOW
            profile: Patient feature profile
            predicted_score: Predicted UPDRS-III score
            
        Returns:
            Clinical narrative string
        """
        narrative = f"Predicted 24-month UPDRS-III score: {predicted_score:.1f}. "
        
        if risk_tier == "HIGH":
            narrative += "High risk of rapid motor progression. Recommendations: "
            narrative += "Consider earlier dopaminergic therapy optimization, "
            narrative += "intensive physical therapy, frequent monitoring (every 3-6 months), "
            narrative += "fall prevention strategies. Reassess DaTSCAN and genetic markers."
        elif risk_tier == "MED":
            narrative += "Moderate motor progression expected. Recommendations: "
            narrative += "Regular monitoring (every 6 months), maintain exercise regimen, "
            narrative += "monitor response to current therapy, consider physical therapy consultation."
        else:
            narrative += "Low risk of rapid progression. Recommendations: "
            narrative += "Standard monitoring, continue current management, "
            narrative += "encourage regular exercise and healthy lifestyle."
        
        return narrative
    
    def _generate_empty_payload(self, patient_id: Optional[str] = None) -> AgentPayload:
        """Generate empty payload for missing data
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            AgentPayload with no data indicators
        """
        return AgentPayload(
            agent_name=self.agent_name,
            timestamp="N/A",
            domain_prediction=0.0,
            uncertainty_variance=1.0,  # Maximum uncertainty
            feature_importance={},
            clinical_narrative="No motor assessment data available for this patient.",
            patient_id=patient_id
        )


if __name__ == "__main__":
    # Test motor agent
    print("Motor Agent Test")
    
    # Example usage
    agent = MotorAgent(data_path="data/raw/formatted_parkinsons_dataset_dataset1.csv")
    agent.load_and_prepare()
    
    # Train model
    agent.train(target_col='updrs3_future_24m', model_type='lightgbm')
    
    # Make prediction
    test_profile = {
        'NUPDRS3_BL': 20.0,
        'months_since_bl': 0.0
    }
    
    payload = agent.analyze(patient_profile=test_profile)
    print(payload)
    print(f"\nNarrative: {payload.clinical_narrative}")
