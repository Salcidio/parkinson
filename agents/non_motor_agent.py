"""
Unified Non-Motor Agent for Parkinson's Disease

Consolidates non-motor symptom analysis across multiple domains:
- Cognitive (MoCA - Montreal Cognitive Assessment)
- Sleep disturbances
- Depression/mood

Merges functionality from:
- agents/non_motor.py
- core_load/non-motor/agent/version1.0.py

Features:
- Multi-domain assessment (sleep, depression, cognitive)
- 24-month progression forecasting per domain
- Domain-specific risk stratification
- SHAP-based interpretability
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

from core.base_agent import ClinicalAgent
from core.protocol import AgentPayload, ModelMetadata


class NonMotorAgent(ClinicalAgent):
    """Specialized agent for non-motor symptom assessment"""
    
    def __init__(
        self,
        domain: str = "cognitive",
        data_path: Optional[str] = None,
        model_path: Optional[str] = None,
        config: Optional[Any] = None
    ):
        """Initialize non-motor agent
        
        Args:
            domain: One of 'cognitive', 'sleep', 'depression'
            data_path: Path to non-motor data
            model_path: Path to saved model
            config: Configuration object
        """
        super().__init__(
            agent_name=f"NonMotorAgent_{domain}",
            data_path=data_path,
            model_path=model_path,
            config=config
        )
        
        self.domain = domain
        
        # Domain-specific configuration
        if config:
            self.max_moca = config.non_motor.max_moca_score
            self.moca_impaired = config.non_motor.moca_impairment_threshold
            self.n_visits_24m = config.data.n_visits_for_24m
        else:
            self.max_moca = 30.0
            self.moca_impaired = 26.0
            self.n_visits_24m = 4
    
    def ingest_ppmi_data(self) -> pd.DataFrame:
        """Load non-motor assessment data
        
        Supports:
        1. Standardized non_motor_merged.csv
        
        Returns:
            DataFrame with non-motor data
        """
        df = pd.read_csv(self.data_path)
        
        # Detect if this is merged file (no header case)
        if 'patient_id' not in df.columns and df.columns[0].startswith('Unnamed'):
            # Merged file format
            column_names = [
                'patient_id', 'assessment_date_x', 'updrs_nonmotor_sleep',
                'assessment_date_y', 'updrs_nonmotor_depression',
                'assessment_date', 'updrs_nonmotor_cognitive'
            ]
            df.columns = column_names
        
        # Standardize column names
        if 'patient_id' in df.columns:
            df = df.rename(columns={'patient_id': 'PATNO'})
        
        # For specific domain files
        domain_col_map = {
            'cognitive': {'MCATOT': 'updrs_nonmotor_cognitive'},
            'sleep': {},  # Already named correctly
            'depression': {}  # Already named correctly
        }
        
        return df
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess non-motor data
        
        Steps:
        1. Parse and consolidate assessment dates
        2. Convert scores to numeric
        3. Calculate months since baseline
        4. Create future targets for progression
        5. Handle domain-specific transformations
        
        Args:
            data: Raw non-motor data
            
        Returns:
            Preprocessed DataFrame
        """
        data = data.copy()
        
        # Consolidate assessment dates
        date_cols = [c for c in data.columns if 'assessment_date' in c.lower() or c == 'INFODT']
        if date_cols:
            # Priority: assessment_date > assessment_date_x > assessment_date_y
            if 'assessment_date' in data.columns:
                data['INFODT'] = pd.to_datetime(data['assessment_date'], errors='coerce')
            elif 'INFODT' in data.columns:
                data['INFODT'] = pd.to_datetime(data['INFODT'], errors='coerce')
            else:
                data['INFODT'] = pd.to_datetime(data[date_cols[0]], errors='coerce')
            
            # Fill NaT with other date columns
            for date_col in date_cols:
                if date_col != 'INFODT' and date_col in data.columns:
                    data['INFODT'] = data['INFODT'].fillna(pd.to_datetime(data[date_col], errors='coerce'))
        
        # Drop rows without dates
        data = data.dropna(subset=['INFODT'])
        data = data.sort_values(['PATNO', 'INFODT'])
        
        # Convert score columns to numeric
        score_cols = ['updrs_nonmotor_sleep', 'updrs_nonmotor_depression', 'updrs_nonmotor_cognitive', 'MCATOT']
        for col in score_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Map MoCA if present
        if 'MCATOT' in data.columns and 'updrs_nonmotor_cognitive' not in data.columns:
            data['updrs_nonmotor_cognitive'] = data['MCATOT']
        
        # For cognitive: invert MoCA (lower = worse, so we want higher severity score)
        if self.domain == 'cognitive' and 'updrs_nonmotor_cognitive' in data.columns:
            # Create severity: (30 - MoCA) / 30, so 0=perfect, 1=worst
            data['cognitive_severity'] = (self.max_moca - data['updrs_nonmotor_cognitive']) / self.max_moca
        
        # Calculate months since baseline
        data['months_since_bl'] = data.groupby('PATNO')['INFODT'].transform(
            lambda x: (x - x.iloc[0]).dt.days / 30.44
        )
        
        # Create future targets
        if 'updrs_nonmotor_sleep' in data.columns:
            data['updrs_nonmotor_sleep_future_24m'] = data.groupby('PATNO')['updrs_nonmotor_sleep'].shift(-self.n_visits_24m)
        if 'updrs_nonmotor_depression' in data.columns:
            data['updrs_nonmotor_depression_future_24m'] = data.groupby('PATNO')['updrs_nonmotor_depression'].shift(-self.n_visits_24m)
        if 'updrs_nonmotor_cognitive' in data.columns:
            data['updrs_nonmotor_cognitive_future_24m'] = data.groupby('PATNO')['updrs_nonmotor_cognitive'].shift(-self.n_visits_24m)
        
        # Define baseline
        self.baseline_df = data.groupby('PATNO').first().reset_index()
        
        return data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer baseline features for progression prediction
        
        Features depend on domain:
        - All domains: baseline scores, months_since_bl
        
        Returns:
            DataFrame with engineered features (indexed by PATNO)
        """
        if self.baseline_df is None:
            raise ValueError("baseline_df not loaded. Run load_and_prepare() first")
        
        feature_cols = ['PATNO', 'months_since_bl']
        
        # Add domain-specific features
        if 'updrs_nonmotor_sleep' in self.baseline_df.columns:
            feature_cols.append('updrs_nonmotor_sleep')
        if 'updrs_nonmotor_depression' in self.baseline_df.columns:
            feature_cols.append('updrs_nonmotor_depression')
        if 'updrs_nonmotor_cognitive' in self.baseline_df.columns:
            feature_cols.append('updrs_nonmotor_cognitive')
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in self.baseline_df.columns]
        
        X = self.baseline_df[feature_cols].copy()
        
        # Rename to indicate baseline
        rename_map = {
            'updrs_nonmotor_sleep': 'updrs_nonmotor_sleep_BL',
            'updrs_nonmotor_depression': 'updrs_nonmotor_depression_BL',
            'updrs_nonmotor_cognitive': 'updrs_nonmotor_cognitive_BL'
        }
        X = X.rename(columns={k: v for k, v in rename_map.items() if k in X.columns})
        X = X.set_index('PATNO')
        
        return X.dropna()
    
    def analyze(self, patient_id: Optional[str] = None, patient_profile: Optional[Dict] = None) -> AgentPayload:
        """Analyze non-motor symptoms for a patient
        
        Two modes:
        1. With patient_id: Load from self.df
        2. With patient_profile: Use provided features
        
        Args:
            patient_id: Patient identifier
            patient_profile: Dictionary of baseline features
            
        Returns:
            AgentPayload with non-motor assessment
        """
        if patient_profile is None and patient_id is None:
            raise ValueError("Must provide either patient_id or patient_profile")
        
        # Mode 1: Load from data
        if patient_id and self.df is not None:
            patient_data = self.df[self.df['PATNO'] == int(patient_id)]
            if patient_data.empty:
                return self._generate_empty_payload(patient_id)
            
            latest = patient_data.iloc[-1]
            
            # Domain-specific processing
            if self.domain == 'cognitive':
                score = latest.get('updrs_nonmotor_cognitive', latest.get('MCATOT', 0))
                severity = (self.max_moca - score) / self.max_moca
                narrative = f"MoCA score: {score:.0f}/{self.max_moca:.0f}. "
                narrative += f"Cognitive status: {'Impaired' if score < self.moca_impaired else 'Normal'}."
                
                # Calculate volatility as uncertainty
                volatility = patient_data['updrs_nonmotor_cognitive'].tail(3).std() if len(patient_data) > 1 else 0.1
                uncertainty = float(np.clip(volatility / self.max_moca, 0.05, 0.4))
                
                feature_importance = {"moca_total": float(score)}
                
            elif self.domain == 'sleep':
                score = latest.get('updrs_nonmotor_sleep', 0)
                # Higher score = worse sleep
                severity = min(score / 15.0, 1.0)  # Normalize (assuming max ~15)
                narrative = f"Sleep disturbance score: {score:.1f}. "
                uncertainty = 0.15
                feature_importance = {"sleep_score": float(score)}
                
            elif self.domain == 'depression':
                score = latest.get('updrs_nonmotor_depression', 0)
                severity = min(score / 20.0, 1.0)  # Normalize (assuming max ~20)
                narrative = f"Depression score: {score:.1f}. "
                uncertainty = 0.15
                feature_importance = {"depression_score": float(score)}
            
            else:
                return self._generate_empty_payload(patient_id)
            
            return AgentPayload(
                agent_name=self.agent_name,
                timestamp=str(latest['INFODT']),
                domain_prediction=float(severity),
                uncertainty_variance=uncertainty,
                feature_importance=feature_importance,
                clinical_narrative=narrative,
                patient_id=patient_id,
                raw_embedding=np.array([score]),
                model_metadata=self.model_metadata
            )
        
        # Mode 2: Use model prediction
        if patient_profile and self.model:
            prediction, feature_importance = self.predict(patient_profile)
            
            risk_tier = self._determine_risk_tier(prediction)
            narrative = self._generate_recommendation(risk_tier, patient_profile, prediction)
            
            uncertainty = 0.15
            
            payload = AgentPayload(
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
                domain_prediction=float(prediction),
                uncertainty_variance=uncertainty,
                feature_importance=feature_importance,
                clinical_narrative=narrative,
                model_metadata=self.model_metadata
            )
            payload.calculate_confidence_interval()
            
            return payload
        
        return self._generate_empty_payload(patient_id)
    
    def _determine_risk_tier(self, prediction: float) -> str:
        """Determine risk tier based on domain and prediction
        
        Args:
            prediction: Predicted score
            
        Returns:
            Risk tier (HIGH/MED/LOW)
        """
        if self.domain == 'sleep':
            return "HIGH" if prediction > 10 else "MED" if prediction > 5 else "LOW"
        elif self.domain == 'depression':
            return "HIGH" if prediction > 15 else "MED" if prediction > 8 else "LOW"
        elif self.domain == 'cognitive':
            # Lower MoCA = worse cognition
            return "HIGH" if prediction < 20 else "MED" if prediction < 25 else "LOW"
        return "UNKNOWN"
    
    def _generate_recommendation(self, tier: str, profile: Dict, prediction: float) -> str:
        """Generate clinical recommendation based on risk tier and domain
        
        Args:
            tier: HIGH, MED, or LOW
            profile: Patient profile
            prediction: Predicted score
            
        Returns:
            Clinical narrative
        """
        narrative = f"Predicted {self.domain} score (24m): {prediction:.1f}. "
        
        if self.domain == 'sleep':
            if tier == "HIGH":
                narrative += "High risk of significant sleep disturbance. Consider polysomnography, evaluate for REM sleep behavior disorder, restless legs. Recommend sleep hygiene, potential pharmacotherapy."
            elif tier == "MED":
                narrative += "Moderate sleep concerns. Advise sleep hygiene, monitor for worsening."
            else:
                narrative += "Low sleep disturbance risk. Continue monitoring."
                
        elif self.domain == 'depression':
            if tier == "HIGH":
                narrative += "High risk of depression. Initiate psychiatric evaluation, consider antidepressant therapy, screen for suicidality."
            elif tier == "MED":
                narrative += "Moderate depressive symptoms. Monitor mood, encourage social engagement."
            else:
                narrative += "Low depressive symptom risk. Routine well-being check."
                
        elif self.domain == 'cognitive':
            if tier == "HIGH":
                narrative += "High risk of cognitive impairment. Recommend neuropsychological assessment, evaluate for dementia, consider cognitive enhancing medications."
            elif tier == "MED":
                narrative += "Moderate cognitive concerns. Encourage cognitive stimulation, regular screening."
            else:
                narrative += "Low cognitive risk. Continue healthy lifestyle."
        
        return narrative
    
    def _generate_empty_payload(self, patient_id: Optional[str] = None) -> AgentPayload:
        """Generate empty payload for missing data"""
        return AgentPayload(
            agent_name=self.agent_name,
            timestamp="N/A",
            domain_prediction=0.0,
            uncertainty_variance=1.0,
            feature_importance={},
            clinical_narrative=f"No {self.domain} assessment data available for this patient.",
            patient_id=patient_id
        )


if __name__ == "__main__":
    # Test non-motor agent
    print("Non-Motor Agent Test - Cognitive Domain")
    
    # Example profile
    test_profile = {
        'updrs_nonmotor_sleep_BL': 7.0,
        'updrs_nonmotor_depression_BL': 10.0,
        'updrs_nonmotor_cognitive_BL': 22.0,
        'months_since_bl': 0.0
    }
    
    agent = NonMotorAgent(domain='cognitive')
    payload = agent.analyze(patient_profile=test_profile)
    print(payload)
    print(f"\nNarrative: {payload.clinical_narrative}")
