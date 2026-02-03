from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import json


@dataclass
class ModelMetadata:
    """Metadata for trained models"""
    model_version: str
    architecture: str  # e.g., "LightGBM", "XGBoost", "Neural Network"
    training_timestamp: str
    training_metrics: Dict[str, float] = field(default_factory=dict)  # MAE, R2, etc.
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class AgentPayload:
    """Standardized output structure for all agents
    
    This protocol ensures consistent communication between agents and the orchestrator.
    All agents must return this structure after analysis/prediction.
    """
    agent_name: str
    timestamp: str
    domain_prediction: float  # normalized severity score [0, 1]
    uncertainty_variance: float  # sigma^2 for inverse variance weighting
    feature_importance: Dict[str, float]  # e.g., {tremor_score: 0.9}
    clinical_narrative: str  # local interpretation
    raw_embedding: Optional[np.ndarray] = field(default=None)  # latent vector
    
    # Extended fields for model provenance
    model_metadata: Optional[ModelMetadata] = field(default=None)
    patient_id: Optional[str] = field(default=None)
    confidence_interval: Optional[tuple] = field(default=None)  # (lower, upper)
    
    def __repr__(self):
        ci_str = ""
        if self.confidence_interval:
            ci_str = f" [{self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f}]"
        return f"<{self.agent_name} | Pred: {self.domain_prediction:.2f} ± {self.uncertainty_variance:.2f}{ci_str}>"
    
    def validate(self) -> bool:
        """Validate payload structure and values"""
        try:
            # Check required fields
            assert self.agent_name and isinstance(self.agent_name, str), "agent_name must be non-empty string"
            assert isinstance(self.domain_prediction, (int, float)), "domain_prediction must be numeric"
            assert isinstance(self.uncertainty_variance, (int, float)), "uncertainty_variance must be numeric"
            assert isinstance(self.feature_importance, dict), "feature_importance must be dict"
            
            # Check value ranges
            assert 0 <= self.domain_prediction <= 1, "domain_prediction should be normalized [0, 1]"
            assert self.uncertainty_variance >= 0, "uncertainty_variance must be non-negative"
            
            # Check feature importance values
            for key, value in self.feature_importance.items():
                assert isinstance(value, (int, float)), f"feature_importance[{key}] must be numeric"
            
            return True
        except AssertionError as e:
            print(f"Validation failed: {e}")
            return False
    
    def to_dict(self, include_embedding: bool = False) -> Dict:
        """Convert to dictionary for JSON serialization
        
        Args:
            include_embedding: Whether to include raw_embedding (can be large)
        
        Returns:
            Dictionary representation
        """
        data = {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "domain_prediction": float(self.domain_prediction),
            "uncertainty_variance": float(self.uncertainty_variance),
            "feature_importance": self.feature_importance,
            "clinical_narrative": self.clinical_narrative,
            "patient_id": self.patient_id,
        }
        
        if self.confidence_interval:
            data["confidence_interval"] = list(self.confidence_interval)
        
        if include_embedding and self.raw_embedding is not None:
            data["raw_embedding"] = self.raw_embedding.tolist()
        
        if self.model_metadata:
            data["model_metadata"] = self.model_metadata.to_dict()
        
        return data
    
    def to_json(self, include_embedding: bool = False) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(include_embedding=include_embedding), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentPayload':
        """Create AgentPayload from dictionary"""
        # Extract and reconstruct nested objects
        model_metadata = None
        if 'model_metadata' in data and data['model_metadata']:
            model_metadata = ModelMetadata.from_dict(data['model_metadata'])
        
        raw_embedding = None
        if 'raw_embedding' in data and data['raw_embedding']:
            raw_embedding = np.array(data['raw_embedding'])
        
        confidence_interval = None
        if 'confidence_interval' in data and data['confidence_interval']:
            confidence_interval = tuple(data['confidence_interval'])
        
        return cls(
            agent_name=data['agent_name'],
            timestamp=data['timestamp'],
            domain_prediction=data['domain_prediction'],
            uncertainty_variance=data['uncertainty_variance'],
            feature_importance=data['feature_importance'],
            clinical_narrative=data['clinical_narrative'],
            raw_embedding=raw_embedding,
            model_metadata=model_metadata,
            patient_id=data.get('patient_id'),
            confidence_interval=confidence_interval
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentPayload':
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def calculate_confidence_interval(self, z_score: float = 1.96):
        """Calculate confidence interval from uncertainty variance
        
        Args:
            z_score: Z-score for confidence level (1.96 for 95% CI)
        """
        std_dev = np.sqrt(self.uncertainty_variance)
        lower = max(0, self.domain_prediction - z_score * std_dev)
        upper = min(1, self.domain_prediction + z_score * std_dev)
        self.confidence_interval = (lower, upper)
        return self.confidence_interval

