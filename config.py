"""
Configuration Management for Parkinson's Multi-Agent System

This module provides centralized configuration for:
- Data paths and directory structure
- Model hyperparameters for all agents
- Training configuration
- Google Colab-specific settings
- Model versioning and export paths
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Project root detection
PROJECT_ROOT = Path(__file__).parent
IS_COLAB = 'COLAB_GPU' in os.environ or '/content/' in str(Path.cwd())

@dataclass
class PathConfig:
    """Path configuration for data and models"""
    # Root directories
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    
    def __post_init__(self):
        if IS_COLAB:
            # Google Colab paths
            self.data_dir = Path('/content/drive/MyDrive/parkinson_data')
            self.models_dir = Path('/content/drive/MyDrive/parkinson_models')
        else:
            # Local paths
            self.data_dir = self.project_root / 'data'
            self.models_dir = self.project_root / 'models'
        
        self.raw_data_dir = self.data_dir / 'raw'
        
    def ensure_dirs(self):
        """Create directories if they don't exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model subdirectories
        for agent in ['motor', 'biomarker', 'non_motor']:
            (self.models_dir / agent).mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Data files and preprocessing configuration"""
    # PPMI data files
    motor_file: str = "MDS_UPDRS_Part_III.csv"
    datscan_file: str = "DaTscan_Analysis.csv"
    moca_file: str = "MoCA.csv"
    non_motor_sleep_file: str = "non_motor_sleep.csv"
    non_motor_depression_file: str = "non_motor_depression.csv"
    non_motor_cognitive_file: str = "non_motor_cognitive.csv"
    
    # Alternative merged files from core_load
    motor_file_alt: str = "formatted_parkinsons_dataset_dataset1.csv"
    datscan_file_alt: str = "datscan.csv"
    merged_non_motor_file: str = "merged_non_motor_data.csv"
    
    # Data preprocessing
    test_size: float = 0.2
    random_state: int = 42
    n_visits_for_24m: int = 4  # Number of visits in 24 months (avg 6 months per visit)
    
    # Missing data strategy
    imputation_method: str = "LOCF"  # Last Observation Carried Forward
    min_baseline_window_months: float = 6.0


@dataclass
class MotorAgentConfig:
    """Motor agent specific configuration"""
    model_type: str = "lightgbm"  # or "xgboost"
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 64
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Target and features
    target_horizon: str = "updrs3_future_24m"
    max_updrs3_score: float = 132.0  # Maximum possible UPDRS-III score
    
    # Risk thresholds
    high_risk_threshold: float = 25.0
    med_risk_threshold: float = 12.0


@dataclass
class BiomarkerAgentConfig:
    """Biomarker agent specific configuration"""
    # SBR thresholds
    healthy_sbr_threshold: float = 2.0
    pd_sbr_threshold: float = 1.5
    severe_sbr_floor: float = 0.5
    healthy_sbr_ceiling: float = 2.5
    
    # Asymmetry threshold
    high_asymmetry_threshold: float = 0.3
    
    # Features
    putamen_cols: List[str] = field(default_factory=lambda: ['PUTAMEN_L', 'PUTAMEN_R'])
    caudate_cols: List[str] = field(default_factory=lambda: ['CAUDATE_L', 'CAUDATE_R'])


@dataclass
class NonMotorAgentConfig:
    """Non-motor agent specific configuration"""
    model_type: str = "lightgbm"
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 64
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Domains
    domains: List[str] = field(default_factory=lambda: ['sleep', 'depression', 'cognitive'])
    
    # MoCA scoring
    max_moca_score: float = 30.0
    moca_impairment_threshold: float = 26.0
    
    # Risk thresholds per domain
    sleep_high_threshold: float = 10.0
    sleep_med_threshold: float = 5.0
    
    depression_high_threshold: float = 15.0
    depression_med_threshold: float = 8.0
    
    cognitive_high_threshold: float = 20.0  # Lower is worse
    cognitive_med_threshold: float = 25.0


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    # Cross-validation
    n_folds: int = 5
    cv_strategy: str = "GroupKFold"  # Patient-grouped CV
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    # Hyperparameter tuning
    use_optuna: bool = False
    optuna_n_trials: int = 50
    
    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = False
    verbose: bool = True
    
    # Model export formats
    export_tensorflow: bool = True
    export_pytorch: bool = True
    export_onnx: bool = True
    export_pickle: bool = True  # Legacy support


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration"""
    # LLM for narrative generation
    llm_provider: str = "ollama"  # or "openai", "anthropic"
    llm_model: str = "mistral"  # or "llama3", "gpt-4", etc.
    
    # Fusion method
    fusion_method: str = "inverse_variance"  # Uncertainty-aware weighting
    epsilon: float = 1e-6  # Avoid division by zero
    
    # Confidence intervals
    confidence_level: float = 0.95  # 95% CI
    z_score: float = 1.96  # For 95% CI


@dataclass
class ColabConfig:
    """Google Colab specific configuration"""
    # Drive mounting
    mount_drive: bool = True
    drive_mount_point: str = "/content/drive"
    
    # GPU/TPU
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None  # MB, None for no limit
    
    # Data download
    download_from_url: bool = False
    data_download_url: Optional[str] = None


class Config:
    """Master configuration object"""
    def __init__(self):
        self.paths = PathConfig()
        self.data = DataConfig()
        self.motor = MotorAgentConfig()
        self.biomarker = BiomarkerAgentConfig()
        self.non_motor = NonMotorAgentConfig()
        self.training = TrainingConfig()
        self.orchestrator = OrchestratorConfig()
        self.colab = ColabConfig()
        
        # Environment detection
        self.is_colab = IS_COLAB
        
    def setup(self):
        """Setup configuration (create directories, etc.)"""
        self.paths.ensure_dirs()
        
        if self.is_colab and self.colab.mount_drive:
            try:
                from google.colab import drive
                drive.mount(self.colab.drive_mount_point)
                print(f"✓ Google Drive mounted at {self.colab.drive_mount_point}")
            except ImportError:
                print("⚠ google.colab not available")
            except Exception as e:
                print(f"⚠ Failed to mount drive: {e}")
        
        print(f"✓ Configuration initialized")
        print(f"  - Environment: {'Google Colab' if self.is_colab else 'Local'}")
        print(f"  - Data directory: {self.paths.data_dir}")
        print(f"  - Models directory: {self.paths.models_dir}")
        
        return self


# Global configuration instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    cfg = Config()
    cfg.setup()
    
    print("\n=== Configuration Summary ===")
    print(f"Project Root: {cfg.paths.project_root}")
    print(f"Data Dir: {cfg.paths.data_dir}")
    print(f"Models Dir: {cfg.paths.models_dir}")
    print(f"Motor Model: {cfg.motor.model_type}")
    print(f"Training Folds: {cfg.training.n_folds}")
