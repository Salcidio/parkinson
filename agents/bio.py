# agents/bio.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .base import ClinicalAgent
from core.protocol import AgentPayload

class BioAgent(ClinicalAgent):
    """
    Specialized agent for Dopamine Transporter (DaT) SPECT imaging.
    Focuses on Striatal Binding Ratios (SBR) from PPMI derived tables.
    """
    
    def ingest_ppmi_data(self) -> pd.DataFrame:
        # PPMI file: DaTscan_Analysis.csv or similar
        df = pd.read_csv(self.data_path)
        
        # Filter for current patient
        patient_df = df[df['PATNO'] == int(self.patient_id)].copy()
        
        # Ensure temporal ordering
        if 'INFODT' in patient_df.columns:
            patient_df['INFODT'] = pd.to_datetime(patient_df['INFODT'])
            patient_df = patient_df.sort_values('INFODT')
            
        return patient_df

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the mean Putamen SBR, which is the most sensitive 
        region for early PD detection.
        """
        # PPMI specific columns for SBR
        regions = ['PUTAMEN_R', 'PUTAMEN_L']
        
        # Drop rows where imaging failed
        data = data.dropna(subset=regions)
        
        # Calculate mean binding ratio
        data['mean_putamen_sbr'] = data[regions].mean(axis=1)
        
        return data

    def analyze(self) -> AgentPayload:
        raw_df = self.ingest_ppmi_data()
        if raw_df.empty:
            # Return empty payload or raise specific custom error
            return self._generate_empty_payload()
            
        clean_df = self.preprocess(raw_df)
        latest_record = clean_df.iloc[-1]
        
        # SBR Logic: Healthy ~2.0+, PD < 1.5. 
        # We invert this for a "Severity Score": 1.0 is severe loss, 0.0 is healthy.
        # Clipping at 2.5 (healthy baseline) and 0.5 (severe floor).
        sbr = latest_record['mean_putamen_sbr']
        severity = np.clip((2.5 - sbr) / (2.5 - 0.5), 0, 1)

        # Uncertainty: Higher if the scan is old or low quality (if QC flag exists)
        # PPMI often has 'SCAN_QUALITY' column.
        quality_uncertainty = 0.1
        if 'SCAN_QUALITY' in latest_record and latest_record['SCAN_QUALITY'] == 0:
            quality_uncertainty = 0.5

        return AgentPayload(
            agent_name="BioAgent_DaT",
            timestamp=str(latest_record['INFODT']),
            domain_prediction=severity,
            uncertainty_variance=quality_uncertainty,
            feature_importance={
                "putamen_left": latest_record['PUTAMEN_L'],
                "putamen_right": latest_record['PUTAMEN_R']
            },
            clinical_narrative=(
                f"Mean Putamen SBR is {sbr:.2f}. "
                f"{'Significant' if sbr < 1.5 else 'Moderate'} dopaminergic deficit observed."
            ),
            raw_embedding=np.array([latest_record['PUTAMEN_L'], latest_record['PUTAMEN_R']])
        )

    def _generate_empty_payload(self):
        # Fallback for patients without imaging data
        return AgentPayload("BioAgent_DaT", "N/A", 0.0, 1.0, {}, "No Imaging Data", None)