# Parkinson's Multi-Agent System

__version__ = "1.0.0"
__author__ = "Parkinson's Research Team"

from config import Config
from core.protocol import AgentPayload, ModelMetadata
from core.base_agent import ClinicalAgent
from agents import MotorAgent, BiomarkerAgent, NonMotorAgent
from orchestrator import ClinicalOrchestrator
from training.pipeline import TrainingPipeline

__all__ = [
    'Config',
    'AgentPayload',
    'ModelMetadata',
    'ClinicalAgent',
    'MotorAgent',
    'BiomarkerAgent',
    'NonMotorAgent',
    'ClinicalOrchestrator',
    'TrainingPipeline'
]
