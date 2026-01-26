import pandas as pd
import numpy as np
import .base import ClinicalAgent
from core.protocol import AgentPayload

class MotorAgent(ClinicalAgent):
    def ingest_ppmi_data(self)->pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df[df['PATNO'] ==int(self.patient_id)]
    
    def preprocess(self, data:pd.DataFrame)->pd.DataFrame
        cols = [c for c in data.columns if 'NP3' in c]
        data['motor_score'] = data[cols].sum(axis=1)
        return data.sort_values('EVENT_ID')


    def analyse(self)->AgentPayload:
        data = self.ingest_ppmi_data()
        clean_data = self.preprocess(data)
        latest_score = clean_data['motor_score'].iloc[-1]
        max_score = 132.0

        normalized_severity = latest_score / max_score
        uncertainty = 0.1 if 'OFF' in clean_data.values else 0.3

        return AgentPayload(
            agent_name="motor_agent",
            timestamp=clean_data['INFODT'].iloc[-1],
            domain_prediction=normalized_severity,
            uncertainty_variance=uncertainty,
            feature_importance={"ridigity":0.4, "tremor":0.6},
            clinical_narrative=f"Patient shows motor score of {latest_score}"
        )


