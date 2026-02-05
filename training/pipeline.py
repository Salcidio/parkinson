"""
Unified Training Pipeline for Parkinson's Multi-Agent System

Orchestrates training of all agents with:
- Sequential or parallel training
- Cross-validation (patient-grouped)
- Hyperparameter tuning
- Model checkpointing
- Metric logging
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import warnings

import pandas as pd
import numpy as np

from agents.motor_agent import MotorAgent
from agents.biomarker_agent import BiomarkerAgent
from agents.non_motor_agent import NonMotorAgent

warnings.filterwarnings('ignore')


class TrainingPipeline:
    """Unified training pipeline for all clinical agents"""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize training pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.agents: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}
        
        if config:
            self.data_dir = config.paths.raw_data_dir
            self.models_dir = config.paths.models_dir
        else:
            self.data_dir = Path("data/raw")
            self.models_dir = Path("models")
    
    def train_motor_agent(
        self,
        data_file: Optional[str] = None,
        model_type: str = "lightgbm",
        **kwargs
    ) -> MotorAgent:
        """Train motor progression agent
        
        Args:
            data_file: Path to motor data CSV
            model_type: "lightgbm" or "xgboost"
            **kwargs: Additional model hyperparameters
            
        Returns:
            Trained MotorAgent
        """
        print("\n=== Training Motor Agent ===")
        
        # Determine data file
        if data_file is None:
            data_file = self.data_dir / self.config.data.motor_file
        
        # Initialize and load data
        agent = MotorAgent(data_path=str(data_file), config=self.config)
        agent.load_and_prepare()
        
        # Train
        model = agent.train(
            target_col='updrs3_future_24m',
            model_type=model_type,
            **kwargs
        )
        
        # Save model
        save_dir = self.models_dir / "motor"
        agent.save_model(save_dir, export_formats=['pickle'])
        
        # Store
        self.agents['motor'] = agent
        self.training_results['motor'] = {
            'metrics': agent.training_metrics,
            'save_dir': str(save_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Motor agent trained and saved to {save_dir}")
        
        return agent
    
    def train_biomarker_agent(
        self,
        data_file: Optional[str] = None,
        **kwargs
    ) -> BiomarkerAgent:
        """Train biomarker agent (DaTSCAN)
        
        Note: Biomarker agent primarily uses rule-based assessment,
        but can be enhanced with predictive models in the future.
        
        Args:
            data_file: Path to DaTSCAN data CSV
            **kwargs: Additional parameters
            
        Returns:
            Initialized BiomarkerAgent
        """
        print("\n=== Initializing Biomarker Agent ===")
        
        # Determine data file
        if data_file is None:
            data_file = self.data_dir / self.config.data.datscan_file
        
        # Initialize and load data
        agent = BiomarkerAgent(data_path=str(data_file), config=self.config)
        agent.load_and_prepare()
        
        # Biomarker agent uses rule-based logic, no training needed
        print("✓ Biomarker agent initialized (rule-based assessment)")
        
        # Store
        self.agents['biomarker'] = agent
        self.training_results['biomarker'] = {
            'type': 'rule-based',
            'timestamp': datetime.now().isoformat()
        }
        
        return agent
    
    def train_non_motor_agent(
        self,
        domain: str = "cognitive",
        data_file: Optional[str] = None,
        model_type: str = "lightgbm",
        **kwargs
    ) -> NonMotorAgent:
        """Train non-motor agent for specific domain
        
        Args:
            domain: 'cognitive', 'sleep', or 'depression'
            data_file: Path to non-motor data CSV
            model_type: "lightgbm" or "xgboost"
            **kwargs: Additional model hyperparameters
            
        Returns:
            Trained NonMotorAgent
        """
        print(f"\n=== Training Non-Motor Agent ({domain}) ===")
        
        # Determine data file
        if data_file is None:
            data_file = self.data_dir / self.config.data.non_motor_file
        
        # Initialize and load data
        agent = NonMotorAgent(domain=domain, data_path=str(data_file), config=self.config)
        agent.load_and_prepare()
        
        # Train (if sufficient data)
        target_col = f'updrs_nonmotor_{domain}_future_24m'
        
        try:
            model = agent.train(
                target_col=target_col,
                model_type=model_type,
                **kwargs
            )
            
            # Save model
            save_dir = self.models_dir / f"non_motor_{domain}"
            agent.save_model(save_dir, export_formats=['pickle'])
            
            self.training_results[f'non_motor_{domain}'] = {
                'metrics': agent.training_metrics,
                'save_dir': str(save_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✓ Non-motor ({domain}) agent trained and saved to {save_dir}")
            
        except ValueError as e:
            print(f"⚠ Could not train non-motor ({domain}) agent: {e}")
            print(f"  Using rule-based assessment instead")
            self.training_results[f'non_motor_{domain}'] = {
                'type': 'rule-based',
                'timestamp': datetime.now().isoformat()
            }
        
        # Store
        self.agents[f'non_motor_{domain}'] = agent
        
        return agent
    
    def train_all(
        self,
        agents_to_train: Optional[List[str]] = None,
        model_type: str = "lightgbm",
        **kwargs
    ) -> Dict[str, Any]:
        """Train all agents
        
        Args:
            agents_to_train: List of agents to train (default: all)
            model_type: Model type for tree-based agents
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of trained agents
        """
        if agents_to_train is None:
            agents_to_train = ['motor', 'biomarker', 'non_motor_cognitive', 
                              'non_motor_sleep', 'non_motor_depression']
        
        print(f"\n{'='*60}")
        print(f"Training Pipeline Started")
        print(f"Agents: {', '.join(agents_to_train)}")
        print(f"Model Type: {model_type}")
        print(f"{'='*60}")
        
        for agent_name in agents_to_train:
            try:
                if agent_name == 'motor':
                    self.train_motor_agent(model_type=model_type, **kwargs)
                    
                elif agent_name == 'biomarker':
                    self.train_biomarker_agent(**kwargs)
                    
                elif agent_name.startswith('non_motor_'):
                    domain = agent_name.replace('non_motor_', '')
                    self.train_non_motor_agent(domain=domain, model_type=model_type, **kwargs)
                    
                else:
                    print(f"⚠ Unknown agent: {agent_name}")
                    
            except Exception as e:
                print(f"✗ Error training {agent_name}: {e}")
                import traceback
                traceback.print_exc()
        
        self._print_summary()
        
        return self.agents
    
    def _print_summary(self):
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        
        for agent_name, results in self.training_results.items():
            print(f"\n{agent_name}:")
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"  MAE: {metrics.get('mae', 'N/A'):.2f}")
                print(f"  R²:  {metrics.get('r2', 'N/A'):.3f}")
                print(f"  RMSE: {metrics.get('rmse', 'N/A'):.2f}")
            else:
                print(f"  Type: {results.get('type', 'Unknown')}")
            print(f"  Timestamp: {results.get('timestamp', 'N/A')}")
        
        print(f"\n{'='*60}")
        print(f"✓ Training Complete - {len(self.agents)} agents ready")
        print(f"{'='*60}\n")
    
    def load_all_agents(self) -> Dict[str, Any]:
        """Load all previously trained agents from disk
        
        Returns:
            Dictionary of loaded agents
        """
        print("Loading trained agents...")
        
        # Motor agent
        motor_dir = self.models_dir / "motor"
        if motor_dir.exists():
            agent = MotorAgent(config=self.config)
            agent.load_model(motor_dir)
            self.agents['motor'] = agent
            print(f"✓ Loaded motor agent from {motor_dir}")
        
        # Biomarker agent (no saved model, just initialized)
        self.agents['biomarker'] = BiomarkerAgent(config=self.config)
        print(f"✓ Initialized biomarker agent")
        
        # Non-motor agents
        for domain in ['cognitive', 'sleep', 'depression']:
            nm_dir = self.models_dir / f"non_motor_{domain}"
            if nm_dir.exists():
                agent = NonMotorAgent(domain=domain, config=self.config)
                agent.load_model(nm_dir)
                self.agents[f'non_motor_{domain}'] = agent
                print(f"✓ Loaded non-motor ({domain}) agent from {nm_dir}")
        
        print(f"\n✓ Loaded {len(self.agents)} agents")
        
        return self.agents


if __name__ == "__main__":
    # Test training pipeline
    print(f"Testing Training Pipeline")
    
    # Create pipeline
    pipeline = TrainingPipeline()
    
    # Train all agents
    agents = pipeline.train_all(
        agents_to_train=['motor'],  # Start with just motor for testing
        model_type='lightgbm'
    )
    
    print("\nDone!")
