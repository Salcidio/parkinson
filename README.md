# Enhanced Parkinson's Multi-Agent System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified multi-agent system for Parkinson's disease progression prediction and clinical assessment, designed for researchers and trainable on Google Colab.

## 🎯 Overview

This system combines three specialized AI agents to provide comprehensive Parkinson's disease assessment:

1. **Motor Agent**: UPDRS-III motor symptom analysis and 24-month progression forecasting
2. **Biomarker Agent**: DaTSCAN imaging analysis (striatal binding ratios)
3. **Non-Motor Agent**: Multi-domain assessment (cognitive, sleep, depression)

All agents use uncertainty-aware predictions that are fused by an orchestrator to generate global risk scores and clinical narratives.

## ✨ Key Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for different clinical domains
- 📊 **SHAP Interpretability**: Explainable predictions using SHAP values
- 🎯 **Risk Stratification**: HIGH/MED/LOW risk tiers with clinical recommendations
- 💾 **Tensor Format Export**: Models saved in TensorFlow/PyTorch formats
- ☁️ **Google Colab Ready**: Full training and inference notebooks included
- 🔬 **Research-Grade**: Validated on PPMI (Parkinson's Progression Markers Initiative) data

## 📁 Project Structure

```
parkinson/
├── config.py                  # Central configuration
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── core/                      # Core framework
│   ├── protocol.py           # AgentPayload & ModelMetadata
│   ├── base_agent.py         # Base agent class
│   └── forecasting.py        # (existing forecasting utilities)
│
├── agents/                    # Agent implementations
│   ├── motor_agent.py        # Motor symptom assessment
│   ├── biomarker_agent.py    # DaTSCAN biomarker analysis
│   └── non_motor_agent.py    # Non-motor symptom assessment
│
├── training/                  # Training infrastructure
│   └── pipeline.py           # Unified training pipeline
│
├── notebooks/                 # Jupyter notebooks
│   ├── train_all_agents.ipynb    # Colab training notebook
│   └── inference_demo.ipynb      # Demonstration notebook
│
├── orchest rator.py            # Multi-agent fusion & reporting
└── main_system.py             # Main system entry point
```

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone <repository-url>
cd parkinson

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

1. Open the training notebook: `notebooks/train_all_agents.ipynb`
2. Upload to Google Colab
3. Run all cells to train agents on your data

## 📚 Usage

### Training Agents

```python
from training.pipeline import TrainingPipeline
from config import Config

# Initialize configuration
cfg = Config()
cfg.setup()

# Create training pipeline
pipeline = TrainingPipeline(config=cfg)

# Train all agents
agents = pipeline.train_all(
    agents_to_train=['motor', 'biomarker', 'non_motor_cognitive'],
    model_type='lightgbm'
)
```

### Making Predictions

```python
from agents import MotorAgent, BiomarkerAgent, NonMotorAgent

# Load trained motor agent
motor_agent = MotorAgent(model_path='models/motor')
motor_agent.load_model('models/motor')

# Make prediction
patient_profile = {
    'NUPDRS3_BL': 20.0,
    'months_since_bl': 0.0
}

payload = motor_agent.analyze(patient_profile=patient_profile)
print(f"Prediction: {payload.domain_prediction:.2f}")
print(f"Narrative: {payload.clinical_narrative}")
```

### Multi-Agent Orchestration

```python
from orchestrator import ClinicalOrchestrator

# Initialize orchestrator
orch = ClinicalOrchestrator()

# Analyze with all agents
motor_payload = motor_agent.analyze(patient_profile=motor_profile)
bio_payload = bio_agent.analyze(patient_profile=bio_profile)
nm_payload = nm_agent.analyze(patient_profile=nm_profile)

# Fuse predictions
fusion_result = orch.uncertainty_aware_fusion([motor_payload, bio_payload, nm_payload])

# Generate clinical report
report = orch.generate_report(fusion_result)
print(report)
```

## 📊 Data Requirements

This system is designed for PPMI (Parkinson's Progression Markers Initiative) data format:

### Required Data Files

- `MDS_UPDRS_Part_III.csv` - Motor assessments
- `DaTscan_Analysis.csv` - DaTSCAN imaging
- `MoCA.csv` - Cognitive assessments

### Data Format

Each file should contain:
- `PATNO`: Patient identifier
- `INFODT`: Assessment date
- Domain-specific columns (UPDRS scores, SBR values, MoCA scores, etc.)

## 🎓 For Researchers

### Model Details

- **Motor Agent**: LightGBM regression, 24-month UPDRS-III prediction
- **Biomarker Agent**: Rule-based SBR assessment with configurable thresholds
- **Non-Motor Agent**: Domain-specific LightGBM models for cognitive, sleep, and depression

### Evaluation Metrics

All models are evaluated using:
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- RMSE (Root Mean Squared Error)
- Patient-grouped cross-validation

### Interpretability

- SHAP (SHapley Additive exPlanations) for all predictions
- Feature importance rankings
- Clinical narratives explaining decisions

## 🔧 Configuration

Edit `config.py` to customize:
- Data paths
- Model hyperparameters
- Risk thresholds
- Training parameters
- Colab-specific settings

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{parkinson_multiagent_2026,
  title={Parkinson's Multi-Agent Progression Prediction System},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- PPMI (Parkinson's Progression Markers Initiative) for data access
- Michael J. Fox Foundation for Parkinson's Research
- Open-source ML community (scikit-learn, LightGBM, SHAP)

## 📞 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This is a research tool. Always consult qualified medical professionals for clinical decisions.
