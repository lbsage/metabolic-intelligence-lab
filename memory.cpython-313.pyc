"""Compatibility entrypoint for the Phase 9 System0 Sandbox.

This module re-exports the core classes used during the iterative sandbox build.
Prefer importing from submodules for production work.
"""

from metabolic_intelligence_lab.core.agent import DUSEAgent
from metabolic_intelligence_lab.core.bus import DUSEBus, EnergyBudget, Message
from metabolic_intelligence_lab.core.memory import GeometricMemory
from metabolic_intelligence_lab.core.planner import LLMStub, RulePlanner, System2Plugin
from metabolic_intelligence_lab.core.policy import PolicyModule
from metabolic_intelligence_lab.core.prospective import ProspectiveEngine, ProspectiveLog
from metabolic_intelligence_lab.core.reward import RewardModel
from metabolic_intelligence_lab.core.salience import SalienceEngine
from metabolic_intelligence_lab.core.tasks import Task, TaskQueue
from metabolic_intelligence_lab.core.telemetry import Telemetry
from metabolic_intelligence_lab.core.ternary_field import TernaryField
from metabolic_intelligence_lab.core.tools import MockToolAPI
from metabolic_intelligence_lab.core.world import WorldState
from metabolic_intelligence_lab.experiments.configs import ExperimentConfig
from metabolic_intelligence_lab.experiments.harness import compare_agents, run_experiments, run_single_config
from metabolic_intelligence_lab.experiments.results import ExperimentResult
