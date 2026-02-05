# main_system.py - Unified Parkinson's Multi-Agent System
from agents.motor_agent import MotorAgent
from agents.biomarker_agent import BiomarkerAgent
from agents.non_motor_agent import NonMotorAgent
from orchestrator import ClinicalOrchestrator
from config import Config
import numpy as np

class PDSystem:
    def __init__(self, patient_id: str, data_dir: str = None, config: Config = None):
        """Initialize Parkinson's Disease Multi-Agent System
        
        Args:
            patient_id: Patient identifier
            data_dir: Directory containing PPMI data files
            config: Configuration object (optional)
        """
        self.patient_id = patient_id
        self.config = config or Config()
        
        # Set data directory
        if data_dir:
            self.config.paths.data_dir = data_dir
            self.config.paths.raw_data_dir = f"{data_dir}/raw" if "/raw" not in data_dir else data_dir
        
        # Initialize specialized agents
        self.agents = [
            MotorAgent(
                data_path=f"{self.config.paths.raw_data_dir}/MDS_UPDRS_Part_III.csv",
                config=self.config
            ),
            BiomarkerAgent(
                data_path=f"{self.config.paths.raw_data_dir}/DaTscan_Analysis.csv",
                config=self.config
            ),
            NonMotorAgent(
                domain="cognitive",
                data_path=f"{self.config.paths.raw_data_dir}/MoCA.csv",
                config=self.config
            )
        ]
        self.orchestrator = ClinicalOrchestrator(config=self.config)

    def run_pipeline(self):
        """Execute full analysis pipeline
        
        Returns:
            Dictionary with global risk score, confidence interval, and clinical summary
        """
        # 1. Decentralized Analysis
        payloads = [agent.analyze(patient_id=self.patient_id) for agent in self.agents]
        
        # 2.Fusion
        fusion_result = self.orchestrator.uncertainty_aware_fusion(payloads)
        
        # 3. Interpretative Narrative (Using local Ollama/Mistral)
        summary = self.orchestrator.generate_report(fusion_result)
        
        # 4.Confidence interval
        ci = self.orchestrator.calculate_confidence_interval(payloads)
        
        return {
            "summary": summary,
            "risk_score": fusion_result['global_risk_score'],
            "confidence_interval": ci,
            "agent_payloads": payloads
        }

if __name__ == "__main__":
    config = Config()
    config.setup()
    
    system = PDSystem(patient_id="3102", data_dir="./data", config=config)
    results = system.run_pipeline()
    
    print(f"Global Severity: {results['risk_score']:.2f} ± {results['confidence_interval']:.2f}")
    print(f"\nClinical Summary:\n{results['summary']}")