"""
Unified Base Agent Class for Parkinson's Multi-Agent System

This module provides the abstract base class that all specialized agents inherit from.
It consolidates common functionality including:
- Data loading and preprocessing
- Model training and evaluation
- SHAP-based explainability
- Model persistence (tensor formats)
- Prediction and payload generation
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import joblib
import json

# Machine learning
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import shap

# Protocol
from core.protocol import AgentPayload, ModelMetadata

warnings.filterwarnings('ignore')


class ClinicalAgent(ABC):
    """Abstract base class for all clinical agents
    
    Provides common infrastructure for:
    - PPMI data ingestion
    - Feature engineering
    - Model training with tree-based models
    - SHAP interpretability
    - Model serialization
    - Prediction and AgentPayload generation
    """
    
    def __init__(
        self,
        agent_name: str,
        data_path: Optional[str] = None,
        model_path: Optional[str] = None,
        config: Optional[Any] = None
    ):
        """Initialize clinical agent
        
        Args:
            agent_name: Unique identifier for this agent
            data_path: Path to data file (can be set later)
            model_path: Path to saved model (for loading)
            config: Configuration object with hyperparameters
        """
        self.agent_name = agent_name
        self.data_path = Path(data_path) if data_path else None
        self.model_path = Path(model_path) if model_path else None
        self.config = config
        
        # Data containers
        self.df: Optional[pd.DataFrame] = None  # Full longitudinal data
        self.baseline_df: Optional[pd.DataFrame] = None  # Baseline features
        
        # Model components
        self.model: Optional[Any] = None  # Trained model
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.model_metadata: Optional[ModelMetadata] = None
        
        # Training metrics
        self.training_metrics: Dict[str, float] = {}
        
    @abstractmethod
    def ingest_ppmi_data(self) -> pd.DataFrame:
        """Load and filter PPMI data for specific patient cohort
        
        Should handle:
        - Reading from CSV
        - Patient filtering
        - Basic column renaming
        
        Returns:
            DataFrame with raw PPMI data
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data specific to clinical domain
        
        Should handle:
        - Missing data imputation (domain-specific strategies)
        - Derived feature calculation
        - Temporal ordering
        - Outlier handling
        
        Args:
            data: Raw data from ingest_ppmi_data
            
        Returns:
            Preprocessed DataFrame
        """
        pass
    
    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        """Engineer baseline features for model training
        
        Should create features from baseline visit(s) that will be used
        to predict future outcomes.
        
        Returns:
            DataFrame with engineered features (indexed by PATNO)
        """
        pass
    
    @abstractmethod
    def analyze(self, patient_id: Optional[str] = None) -> AgentPayload:
        """Execute analysis for a patient and return standardized payload
        
        This is the main entry point for agent execution.
        
        Args:
            patient_id: Optional patient ID (if None, use loaded data)
            
        Returns:
            AgentPayload with predictions and interpretations
        """
        pass
    
    def load_and_prepare(self, data_path: Optional[str] = None) -> 'ClinicalAgent':
        """Convenience method to load and preprocess data
        
        Args:
            data_path: Path to data file (overrides instance path)
            
        Returns:
            Self for method chaining
        """
        if data_path:
            self.data_path = Path(data_path)
        
        if not self.data_path or not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load and preprocess
        raw_data = self.ingest_ppmi_data()
        self.df = self.preprocess(raw_data)
        
        print(f"✓ {self.agent_name}: Loaded {len(self.df)} records for {self.df['PATNO'].nunique()} patients")
        
        return self
    
    def train(
        self,
        target_col: str,
        model_type: str = "lightgbm",
        test_size: float = 0.2,
        random_state: int = 42,
        **model_kwargs
    ) -> Any:
        """Train progression model
        
        Args:
            target_col: Name of target column in df
            model_type: "lightgbm" or "xgboost"
            test_size: Fraction of data for testing
            random_state: Random seed
            **model_kwargs: Additional model hyperparameters
            
        Returns:
            Trained model
        """
        # Engineer features
        X = self.engineer_features()
        self.feature_names = list(X.columns)
        
        # Align target
        target_series = self.df.set_index('PATNO').groupby(level=0)[target_col].first()
        X_aligned, y_aligned = X.align(target_series, join='inner', axis=0)
        y_aligned = y_aligned.dropna()
        X_aligned = X_aligned.loc[y_aligned.index]
        
        if X_aligned.empty or y_aligned.empty:
            raise ValueError(f"Insufficient data for training '{target_col}'")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_aligned, y_aligned, test_size=test_size, random_state=random_state
        )
        
        # Default hyperparameters
        default_params = {
            'n_estimators': 800,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        default_params.update(model_kwargs)
        
        # Train model
        if model_type == "lightgbm":
            default_params['num_leaves'] = default_params.get('num_leaves', 64)
            model = lgb.LGBMRegressor(**default_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        elif model_type == "xgboost":
            default_params['objective'] = 'reg:squarederror'
            model = xgb.XGBRegressor(**default_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.training_metrics = {
            'mae': float(mae),
            'r2': float(r2),
            'rmse': float(rmse),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"✓ {self.agent_name} Training Complete:")
        print(f"  MAE: {mae:.2f} | R²: {r2:.3f} | RMSE: {rmse:.2f}")
        
        # Setup SHAP
        self.model = model
        self.shap_explainer = shap.TreeExplainer(model)
        
        # Create metadata
        self.model_metadata = ModelMetadata(
            model_version="1.0",
            architecture=model_type,
            training_timestamp=datetime.now().isoformat(),
            training_metrics=self.training_metrics,
            hyperparameters=default_params,
            feature_names=self.feature_names
        )
        
        return model
    
    def predict(self, patient_profile: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Make prediction for a patient profile
        
        Args:
            patient_profile: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, feature_importances)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([patient_profile])
        
        # Align features
        if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
            expected_features = self.model.feature_names_in_
        else:
            expected_features = self.feature_names
        
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[expected_features]
        
        # Predict
        prediction = float(self.model.predict(input_df)[0])
        
        # SHAP explanation
        feature_importances = self._compute_shap_importances(input_df)
        
        return prediction, feature_importances
    
    def _compute_shap_importances(self, input_df: pd.DataFrame) -> Dict[str, float]:
        """Compute SHAP-based feature importances
        
        Args:
            input_df: Input features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.shap_explainer is None:
            # Fallback to model feature importances
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.feature_names or input_df.columns
                return dict(zip(feature_names, map(float, importances)))
            return {}
        
        try:
            shap_values = self.shap_explainer(input_df)
            shap_vals_single = shap_values.values[0] if len(shap_values.values.shape) > 1 else shap_values.values
            
            feature_names = input_df.columns
            return {name: float(abs(val)) for name, val in zip(feature_names, shap_vals_single)}
        except Exception as e:
            print(f"Warning: SHAP computation failed: {e}")
            return {}
    
    def save_model(self, save_dir: Path, export_formats: List[str] = ['pickle', 'onnx']):
        """Save model in multiple formats
        
        Args:
            save_dir: Directory to save model
            export_formats: List of formats ('pickle', 'onnx', 'tensorflow', 'pytorch')
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Pickle (legacy)
        if 'pickle' in export_formats:
            model_file = save_dir / f"{self.agent_name}_model.pkl"
            joblib.dump(self.model, model_file)
            print(f"✓ Saved model (pickle): {model_file}")
        
        # Metadata
        metadata_file = save_dir / f"{self.agent_name}_metadata.json"
        if self.model_metadata:
            with open(metadata_file, 'w') as f:
                json.dump(self.model_metadata.to_dict(), f, indent=2)
            print(f"✓ Saved metadata: {metadata_file}")
        
        # ONNX export (future enhancement)
        if 'onnx' in export_formats:
            print(f"⚠ ONNX export not yet implemented")
        
        # TensorFlow/PyTorch (future enhancement)
        if 'tensorflow' in export_formats or 'pytorch' in export_formats:
            print(f"⚠ TensorFlow/PyTorch export not yet implemented")
    
    def load_model(self, load_dir: Path):
        """Load model from disk
        
        Args:
            load_dir: Directory containing saved model
        """
        load_dir = Path(load_dir)
        
        # Load pickle model
        model_file = load_dir / f"{self.agent_name}_model.pkl"
        if model_file.exists():
            self.model = joblib.load(model_file)
            print(f"✓ Loaded model: {model_file}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load metadata
        metadata_file = load_dir / f"{self.agent_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                self.model_metadata = ModelMetadata.from_dict(metadata_dict)
                self.feature_names = self.model_metadata.feature_names
            print(f"✓ Loaded metadata: {metadata_file}")
        
        # Setup SHAP
        if self.model:
            self.shap_explainer = shap.TreeExplainer(self.model)
    
    def __repr__(self):
        status = "trained" if self.model else "untrained"
        data_status = f"{len(self.df)} records" if self.df is not None else "no data"
        return f"<{self.agent_name} ({status}, {data_status})>"
