"""
Unified Biomarker Agent for Parkinson's Disease

Consolidates DaTSCAN (Dopamine Transporter SPECT) imaging biomarker analysis
Merges functionality from:
- agents/bio.py
- core_load/biomarker/agents/version1.py

Features:
- Striatal Binding Ratio (SBR) analysis (putamen, caudate)
- Dopaminergic deficit severity assessment
- Risk stratification (HIGH/MED/LOW)
- Asymmetry calculations
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from core.base_agent import ClinicalAgent
from core.protocol import AgentPayload, ModelMetadata


class BiomarkerAgent(ClinicalAgent):
    """Specialized agent for DaTSCAN biomarker assessment"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        model_path: Optional[str] = None,
        config: Optional[Any] = None
    ):
        super().__init__(
            agent_name="BiomarkerAgent_DaT",
            data_path=data_path,
            model_path=model_path,
            config=config
        )
        
        # Biomarker-specific configuration
        if config:
            self.healthy_sbr = config.biomarker.healthy_sbr_threshold
            self.pd_sbr = config.biomarker.pd_sbr_threshold
            self.severe_floor = config.biomarker.severe_sbr_floor
            self.healthy_ceiling = config.biomarker.healthy_sbr_ceiling
            self.high_asymmetry = config.biomarker.high_asymmetry_threshold
        else:
            self.healthy_sbr = 2.0
            self.pd_sbr = 1.5
            self.severe_floor = 0.5
            self.healthy_ceiling = 2.5
            self.high_asymmetry = 0.3
    
    def ingest_ppmi_data(self) -> pd.DataFrame:
        """Load DaTSCAN imaging data
        
        Supports multiple file formats:
        1. Standard PPMI DaTscan_Analysis.csv
        2. Preprocessed datscan.csv
        
        Returns:
            DataFrame with DaTSCAN SBR data
        """
        df = pd.read_csv(self.data_path)
        
        # Detect format and standardize column names
        if 'patient_id' in df.columns:
            df = df.rename(columns={
                'patient_id': 'PATNO',
                'assessment_date': 'INFODT',
                'datscan_left_putamen': 'PUTAMEN_L',
                'datscan_right_putamen': 'PUTAMEN_R',
                'datscan_left_caudate': 'CAUDATE_L',
                'datscan_right_caudate': 'CAUDATE_R'
            })
        
        # Ensure required columns
        required_cols = ['PATNO', 'INFODT', 'PUTAMEN_L', 'PUTAMEN_R']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DaTSCAN data
        
        Steps:
        1. Parse dates and sort temporally
        2. Calculate mean SBR values (putamen, caudate)
        3. Calculate asymmetry metrics
        4. Define baseline imaging
        5. Create binary risk indicators
        
        Args:
            data: Raw DaTSCAN data
            
        Returns:
            Preprocessed DataFrame
        """
        data = data.copy()
        
        # Parse dates
        data['INFODT'] = pd.to_datetime(data['INFODT'], errors='coerce')
        data = data.dropna(subset=['INFODT', 'PUTAMEN_L', 'PUTAMEN_R'])
        data = data.sort_values(['PATNO', 'INFODT'])
        
        # Calculate months since baseline
        data['months_since_bl'] = data.groupby('PATNO')['INFODT'].transform(
            lambda x: (x - x.iloc[0]).dt.days / 30.44
        )
        
        # Calculate mean SBRs
        data['putamen_mean_sbr'] = (data['PUTAMEN_L'] + data['PUTAMEN_R']) / 2
        
        if 'CAUDATE_L' in data.columns and 'CAUDATE_R' in data.columns:
            data['caudate_mean_sbr'] = (data['CAUDATE_L'] + data['CAUDATE_R']) / 2
        
        # Calculate asymmetry
        data['striatal_asym'] = abs(data['PUTAMEN_L'] - data['PUTAMEN_R'])
        
        # Binary risk indicator (low SBR = high risk)
        data['low_dat_risk'] = (data['putamen_mean_sbr'] < self.healthy_sbr).astype(int)
        
        # Define baseline (earliest scan within 6 months)
        baseline = data[data['months_since_bl'] <= 6].groupby('PATNO').first().reset_index()
        self.baseline_df = baseline
        
        return data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer baseline DaTSCAN features
        
        Features:
        - Putamen mean SBR
        - Caudate mean SBR (if available)
        - Striatal asymmetry
        - Low DaT risk binary indicator
        
        Returns:
            DataFrame with engineered features (indexed by PATNO)
        """
        if self.baseline_df is None:
            raise ValueError("baseline_df not loaded. Run load_and_prepare() first")
        
        feature_cols = ['PATNO', 'putamen_mean_sbr', 'striatal_asym', 'low_dat_risk']
        
        if 'caudate_mean_sbr' in self.baseline_df.columns:
            feature_cols.insert(2, 'caudate_mean_sbr')
        
        X = self.baseline_df[feature_cols].copy()
        X = X.set_index('PATNO')
        
        return X.fillna(0)  # Fill NaNs with 0
    
    def analyze(self, patient_id: Optional[str] = None, patient_profile: Optional[Dict] = None) -> AgentPayload:
        """Analyze DaTSCAN biomarkers for a patient
        
        Two modes:
        1. With patient_id: Load from self.df
        2. With patient_profile: Use provided features
        
        Args:
            patient_id: Patient identifier
            patient_profile: Dictionary of biomarker features
            
        Returns:
            AgentPayload with biomarker assessment
        """
        if patient_profile is None and patient_id is None:
            raise ValueError("Must provide either patient_id or patient_profile")
        
        # Mode 1: Load from data
        if patient_id and self.df is not None:
            patient_data = self.df[self.df['PATNO'] == int(patient_id)]
            if patient_data.empty:
                return self._generate_empty_payload(patient_id)
            
            latest = patient_data.iloc[-1]
            sbr = latest['putamen_mean_sbr']
            
            # Calculate severity score (inverted SBR)
            # Healthy (~2.5) = 0, Severe (<0.5) = 1
            severity = np.clip((self.healthy_ceiling - sbr) / (self.healthy_ceiling - self.severe_floor), 0, 1)
            
            # Uncertainty based on scan quality (if available)
            quality_uncertainty = 0.1
            if 'SCAN_QUALITY' in latest and latest['SCAN_QUALITY'] == 0:
                quality_uncertainty = 0.5
            
            # Feature importance
            feature_importance = {
                "putamen_left": float(latest['PUTAMEN_L']),
                "putamen_right": float(latest['PUTAMEN_R']),
                "putamen_mean_sbr": float(sbr)
            }
            
            if 'CAUDATE_L' in latest:
                feature_importance["caudate_left"] = float(latest['CAUDATE_L'])
                feature_importance["caudate_right"] = float(latest['CAUDATE_R'])
            
            # Clinical narrative
            narrative = f"Mean Putamen SBR is {sbr:.2f}. "
            if sbr < self.pd_sbr:
                narrative += "Significant dopaminergic deficit observed. "
            elif sbr < self.healthy_sbr:
                narrative += "Moderate dopaminergic deficit. "
            else:
                narrative += "Normal dopaminergic function. "
            
            if latest['striatal_asym'] > self.high_asymmetry:
                narrative += f"Notable striatal asymmetry ({latest['striatal_asym']:.2f})."
            
            return AgentPayload(
                agent_name=self.agent_name,
                timestamp=str(latest['INFODT']),
                domain_prediction=float(severity),
                uncertainty_variance=quality_uncertainty,
                feature_importance=feature_importance,
                clinical_narrative=narrative,
                patient_id=patient_id,
                raw_embedding=np.array([latest['PUTAMEN_L'], latest['PUTAMEN_R']]),
                model_metadata=self.model_metadata
            )
        
        # Mode 2: Use profile for risk assessment
        if patient_profile:
            sbr = patient_profile.get('putamen_mean_sbr', 2.0)
            low_dat_risk = patient_profile.get('low_dat_risk', 0)
            striatal_asym = patient_profile.get('striatal_asym', 0)
            
            # Severity score
            severity = np.clip((self.healthy_ceiling - sbr) / (self.healthy_ceiling - self.severe_floor), 0, 1)
            
            # Risk tier determination
            risk_tier = self._determine_risk_tier(patient_profile)
            narrative = self._generate_recommendation(risk_tier, patient_profile)
            
            # Feature importance (heuristic)
            feature_importance = {
                'putamen_mean_sbr': 0.9,  # Highest importance
                'low_dat_risk': 0.8,
                'striatal_asym': 0.3
            }
            if 'caudate_mean_sbr' in patient_profile:
                feature_importance['caudate_mean_sbr'] = 0.2
            
            uncertainty = 0.15
            
            payload = AgentPayload(
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
                domain_prediction=float(severity),
                uncertainty_variance=uncertainty,
                feature_importance=feature_importance,
                clinical_narrative=narrative,
                model_metadata=self.model_metadata
            )
            payload.calculate_confidence_interval()
            
            return payload
        
        return self._generate_empty_payload(patient_id)
    
    def _determine_risk_tier(self, profile: Dict) -> str:
        """Determine risk tier based on DaTSCAN features
        
        Args:
            profile: Biomarker feature dictionary
            
        Returns:
            Risk tier (HIGH/MED/LOW)
        """
        low_dat_risk = profile.get('low_dat_risk', 0)
        striatal_asym = profile.get('striatal_asym', 0)
        caudate_sbr = profile.get('caudate_mean_sbr', 99)
        
        if low_dat_risk == 1:
            return "HIGH"  # Low putamen SBR
        elif striatal_asym > self.high_asymmetry or caudate_sbr < self.pd_sbr:
            return "MED"  # Moderate asymmetry or lower caudate
        else:
            return "LOW"  # Normal
    
    def _generate_recommendation(self, tier: str, profile: Dict) -> str:
        """Generate clinical recommendation based on risk tier
        
        Args:
            tier: HIGH, MED, or LOW
            profile: Biomarker profile
            
        Returns:
            Clinical narrative
        """
        sbr = profile.get('putamen_mean_sbr', 0)
        narrative = f"Putamen SBR: {sbr:.2f}. "
        
        if tier == "HIGH":
            narrative += "High risk: Significant dopaminergic deficit (low putamen SBR). "
            narrative += "Recommendations: Confirm with clinical symptoms, consider dopaminergic therapy, close monitoring for motor progression."
            if profile.get('striatal_asym', 0) > self.high_asymmetry:
                narrative += " Significant striatal asymmetry noted."
        elif tier == "MED":
            narrative += "Moderate risk: Some dopaminergic deficit or asymmetry. "
            narrative += "Recommendations: Continued clinical monitoring and reassessment of symptoms."
        else:
            narrative += "Low risk: Normal DaTSCAN findings. "
            narrative += "Suggests intact presynaptic dopaminergic neurons. "
            narrative += "Clinical symptoms may indicate essential tremor or another condition. Routine monitoring as appropriate."
        
        return narrative
    
    def _generate_empty_payload(self, patient_id: Optional[str] = None) -> AgentPayload:
        """Generate empty payload for missing data"""
        return AgentPayload(
            agent_name=self.agent_name,
            timestamp="N/A",
            domain_prediction=0.0,
            uncertainty_variance=1.0,
            feature_importance={},
            clinical_narrative="No DaTSCAN imaging data available for this patient.",
            patient_id=patient_id
        )


if __name__ == "__main__":
    # Test biomarker agent
    print("Biomarker Agent Test")
    
    # Example profile
    test_profile = {
        'putamen_mean_sbr': 1.8,
        'caudate_mean_sbr': 2.1,
        'striatal_asym': 0.4,
        'low_dat_risk': 1
    }
    
    agent = BiomarkerAgent()
    payload = agent.analyze(patient_profile=test_profile)
    print(payload)
    print(f"\nNarrative: {payload.clinical_narrative}")
