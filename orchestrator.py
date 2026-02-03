# orchestrator.py - Enhanced Clinical Orchestrator
from typing import List, Optional, Any
import numpy as np

from core.protocol import AgentPayload
from langchain_community.llms import Ollama 
# NOTE: Use Ollama with "mistral" or "llama3" for FREE local inference.

class ClinicalOrchestrator:
    def __init__(self, config: Optional[Any] = None):
        """Initialize clinical orchestrator
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        
        # LLM configuration
        llm_model = "mistral" if not config else config.orchestrator.llm_model
        
        # Initialize local free LLM
        try:
            self.narrator = Ollama(model=llm_model)
        except Exception as e:
            print(f"Warning: Could not initialize Ollama LLM: {e}")
            print("Clinical narratives will use template-based generation")
            self.narrator = None

    def uncertainty_aware_fusion(self, payloads: List[AgentPayload]) -> dict:
        """Inverse Variance Weighting (IVW) for fusion.
        
        Agents with higher uncertainty contribute less to the global forecast.
        
        Args:
            payloads: List of AgentPayload objects from different agents
            
        Returns:
            Dictionary with global risk score, context, and agent data
        """
        if not payloads:
            return {
                "global_risk_score": 0.0,
                "context_blob": "No agent data available",
                "agent_data": []
            }
        
        numerator = 0
        denominator = 0
        epsilon = 1e-6 if not self.config else self.config.orchestrator.epsilon
        
        for p in payloads:
            weight = 1 / (p.uncertainty_variance + epsilon) # Avoid div by zero
            numerator += p.domain_prediction * weight
            denominator += weight
            
        global_risk = numerator / denominator if denominator > 0 else 0.0
        
        # Combine narratives
        context = "\n".join([f"{p.agent_name}: {p.clinical_narrative}" for p in payloads])
        
        return {
            "global_risk_score": global_risk,
            "context_blob": context,
            "agent_data": payloads
        }
    
    def calculate_confidence_interval(
        self,
        payloads: List[AgentPayload],
        confidence_level: float = 0.95
    ) -> float:
        """Calculate confidence interval for global risk score
        
        Uses inverse variance weighting to compute pooled uncertainty.
        
        Args:
            payloads: List of AgentPayload objects
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Half-width of confidence interval
        """
        if not payloads:
            return 1.0  # Maximum uncertainty
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(confidence_level, 1.96)
        
        epsilon = 1e-6 if not self.config else self.config.orchestrator.epsilon
        
        # Pooled variance using inverse variance weighting
        pooled_variance = 1 / sum(1 / (p.uncertainty_variance + epsilon) for p in payloads)
        
        # Confidence interval half-width
        ci_half_width = z_score * np.sqrt(pooled_variance)
        
        return float(ci_half_width)

    def generate_report(self, fusion_result: dict) -> str:
        """Generate clinical summary report
        
        Uses LLM if available, otherwise falls back to template-based generation.
        
        Args:
            fusion_result: Fusion result from uncertainty_aware_fusion
            
        Returns:
            Clinical narrative string
        """
        if self.narrator:
            # LLM-based generation
            try:
                prompt = (
                    f"Context:\n{fusion_result['context_blob']}\n"
                    f"Global Risk Score: {fusion_result['global_risk_score']:.2f}\n\n"
                    "Task: Act as a senior neurologist. Synthesize the above into a clinical summary "
                    "explaining the progression logic. Be concise and actionable."
                )
                return self.narrator.invoke(prompt)
            except Exception as e:
                print(f"Warning: LLM generation failed: {e}")
                # Fall through to template
        
        # Template-based generation (fallback)
        return self._generate_template_report(fusion_result)
    
    def _generate_template_report(self, fusion_result: dict) -> str:
        """Generate template-based clinical report
        
        Args:
            fusion_result: Fusion result dictionary
            
        Returns:
            Clinical narrative string
        """
        global_risk = fusion_result['global_risk_score']
        payloads = fusion_result['agent_data']
        
        # Risk tier
        if global_risk > 0.7:
            tier = "HIGH"
        elif global_risk > 0.4:
            tier = "MODERATE"
        else:
            tier = "LOW"
        
        report = f"**Clinical Assessment Summary**\n\n"
        report += f"Global Risk Score: {global_risk:.2f} ({tier} risk)\n\n"
        
        # Agent-specific findings
        report += "**Agent Findings:**\n"
        for payload in payloads:
            report += f"- {payload.agent_name}: {payload.clinical_narrative}\n"
        
        # Overall recommendation
        report += f"\n**Recommendation**: "
        if tier == "HIGH":
            report += "Close monitoring recommended. Consider therapy optimization and frequent follow-up."
        elif tier == "MODERATE":
            report += "Regular monitoring advised. Continue current management with periodic reassessment."
        else:
            report += "Standard care appropriate. Continue routine monitoring."
        
        return report


# Usage Example
if __name__ == "__main__":
    from agents.motor_agent import MotorAgent
    
    # 1. Instantiate Agents
    motor_agent = MotorAgent(data_path="data/mock_updrs.csv")
    
    # 2. Run Analysis
    payloads = [motor_agent.analyze(patient_id="3102")] # Add Bio/Non-Motor here
    
    # 3. Fuse and Report
    orch = ClinicalOrchestrator()
    fusion = orch.uncertainty_aware_fusion(payloads)
    ci = orch.calculate_confidence_interval(payloads)
    report = orch.generate_report(fusion)
    
    print("--- Clinical Forecast ---")
    print(f"Risk: {fusion['global_risk_score']:.2f} ± {ci:.2f}")
    print(f"\n{report}")