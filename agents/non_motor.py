# agents/non_motor.py
import pandas as pd
import numpy as np
from .base import ClinicalAgent
from core.protocol import AgentPayload

class NonMotorAgent(ClinicalAgent):
    def ingest_ppmi_data(self) -> pd.DataFrame:
        # Target: MoCA (Montreal Cognitive Assessment)
        df = pd.read_csv(self.data_path)
        return df[df['PATNO'] == int(self.patient_id)].sort_values('INFODT')

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Focus on 'MCATOT' column (Total MoCA Score, 0-30)
        data = data.dropna(subset=['MCATOT'])
        # Invert: Lower MoCA = Higher Severity
        data['severity'] = (30 - data['MCATOT']) / 30
        return data

    def analyze(self) -> AgentPayload:
        df = self.preprocess(self.ingest_ppmi_data())
        if df.empty: return self._empty_payload()

        latest = df.iloc[-1]
        # Calculate cognitive volatility (standard deviation of last 3 visits)
        volatility = df['severity'].tail(3).std() if len(df) > 1 else 0.1

        return AgentPayload(
            agent_name="NonMotor_Cognitive",
            timestamp=str(latest['INFODT']),
            domain_prediction=latest['severity'],
            uncertainty_variance=float(np.clip(volatility, 0.05, 0.4)),
            feature_importance={"moca_total": latest['MCATOT']},
            clinical_narrative=f"MoCA score: {latest['MCATOT']}/30. Cognitive status: {'Impaired' if latest['MCATOT'] < 26 else 'Normal'}.",
            raw_embedding=np.array([latest['MCATOT']])
        )

    def _empty_payload(self):
        return AgentPayload("NonMotor_Cognitive", "N/A", 0.0, 1.0, {}, "No data", None)