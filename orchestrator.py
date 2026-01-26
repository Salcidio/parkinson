# orchestrator.py
from typing import List
from core.protocol import AgentPayload
from langchain_community.llms import Ollama 
# NOTE: Use Ollama with "mistral" or "llama3" for FREE local inference.

class ClinicalOrchestrator:
    def __init__(self):
        # Initialize local free LLM
        self.narrator = Ollama(model="mistral") 

    def uncertainty_aware_fusion(self, payloads: List[AgentPayload]) -> dict:
        """
        Inverse Variance Weighting (IVW) for fusion.
        Agents with higher uncertainty contribute less to the global forecast.
        """
        numerator = 0
        denominator = 0
        
        for p in payloads:
            weight = 1 / (p.uncertainty_variance + 1e-6) # Avoid div by zero
            numerator += p.domain_prediction * weight
            denominator += weight
            
        global_risk = numerator / denominator
        
        # Combine narratives
        context = "\n".join([f"{p.agent_name}: {p.clinical_narrative}" for p in payloads])
        
        return {
            "global_risk_score": global_risk,
            "context_blob": context,
            "agent_data": payloads
        }

    def generate_report(self, fusion_result: dict) -> str:
        prompt = (
            f"Context:\n{fusion_result['context_blob']}\n"
            f"Global Risk Score: {fusion_result['global_risk_score']:.2f}\n\n"
            "Task: Act as a senior neurologist. Synthesize the above into a clinical summary "
            "explaining the progression logic. Be concise."
        )
        # Invoking local LLM
        return self.narrator.invoke(prompt)

# Usage Example
if __name__ == "__main__":
    # 1. Instantiate Agents
    motor_agent = MotorAgent(patient_id="3102", data_path="data/mock_updrs.csv")
    
    # 2. Run Analysis
    payloads = [motor_agent.analyze()] # Add Bio/Non-Motor here
    
    # 3. Fuse and Report
    orch = ClinicalOrchestrator()
    fusion = orch.uncertainty_aware_fusion(payloads)
    report = orch.generate_report(fusion)
    
    print("--- Clinical Forecast ---")
    print(report)