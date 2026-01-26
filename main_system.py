# main_system.py
from agents.motor import MotorAgent
from agents.bio import BioAgent
from agents.non_motor import NonMotorAgent
from orchestrator import ClinicalOrchestrator

class PDSystem:
    def __init__(self, patient_id: str, data_dir: str):
        self.patient_id = patient_id
        # Initialize specialized agents
        self.agents = [
            MotorAgent(patient_id, f"{data_dir}/MDS_UPDRS_Part_III.csv"),
            BioAgent(patient_id, f"{data_dir}/DaTscan_Analysis.csv"),
            NonMotorAgent(patient_id, f"{data_dir}/MoCA.csv")
        ]
        self.orchestrator = ClinicalOrchestrator()

    def run_pipeline(self):
        # 1. Decentralized Analysis
        payloads = [agent.analyze() for agent in self.agents]
        
        # 2. Weighted Fusion
        # Logic: Global Score = Σ (Pred_i * (1/Var_i)) / Σ (1/Var_i)
        fusion_result = self.orchestrator.uncertainty_aware_fusion(payloads)
        
        # 3. Interpretative Narrative (Using local Ollama/Mistral)
        summary = self.orchestrator.generate_report(fusion_result)
        
        return {
            "summary": summary,
            "risk_score": fusion_result['global_risk_score'],
            "confidence_interval": 1.96 * np.sqrt(1 / sum(1/p.uncertainty_variance for p in payloads))
        }

if __name__ == "__main__":
    system = PDSystem(patient_id="3102", data_dir="./ppmi_data")
    results = system.run_pipeline()
    
    print(f"Global Severity: {results['risk_score']:.2f} ± {results['confidence_interval']:.2f}")
    print(f"Summary: {results['summary']}")