from abc import ABC, abstractmethod
import pandas as pd
from core.protocol import AgentPayload

class ClinicalAgent(ABC):
    def __init__(self, patient_id: str, data_path:str):
        self.patient_id = patient_id
        self.data_path = data_path
        self.model = None # Placeholder for sklearn/torch model

        @abstractmethod
        def ingest_ppmi_data(self)->pd.DataFrame:
            """load specific ppmi csv files and filter based on patient_id"""
            pass
        
        @abstractmethod
        def preprocess(self, data:pd.DataFrame)->pd.DataFrame:
            """handle missingness specific to domain (e.g LOCF for UPDRS)"""
            pass
        
        @abstractmethod
        def analyse(self)-> AgentPayload:
            """run inference and return structured payload"""
            passn





