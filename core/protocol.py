from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np



@dataclass
class AgentPayload:
    """Standardized output structure for all agents"""
    agent_name: str
    timestamp: str
    domain_prediction: float # normalized severity score
    uncertainty_variance: float #sigma 2
    feature_importance: Dict[str, float] # e.g {tremor_score: 0.9}
    clinical_narrative: str # local interpretation
    raw_embedding: Optional[np.ndarray] = field(default=None) # latent vector

    def __repr__(self):
        return f"<{self.agent_name} | Pred: {self.domain_prediction:.2f} +/- {self.uncertainty_variance:.2f}>"
        
