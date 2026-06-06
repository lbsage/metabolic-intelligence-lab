"""
metabolic-intelligence-lab public API.

Import the most-used classes directly from this package:

    from metabolic_intelligence_lab import System0Sandbox, ExperimentConfig
"""
from metabolic_intelligence_lab.system0_sandbox import (
    System0Sandbox,
    evolve_world,
    world_to_context,
)
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
from metabolic_intelligence_lab.experiments.harness import (
    compare_agents,
    run_experiments,
    run_single_config,
)
from metabolic_intelligence_lab.experiments.results import ExperimentResult, summarize_rows
from metabolic_intelligence_lab.persistence import (
    RunSnapshot,
    latest_run,
    list_runs,
    load_snapshot,
    save_snapshot,
)

__all__ = [
    # sandbox
    "System0Sandbox",
    "evolve_world",
    "world_to_context",
    # core
    "DUSEAgent",
    "DUSEBus",
    "EnergyBudget",
    "GeometricMemory",
    "LLMStub",
    "Message",
    "MockToolAPI",
    "PolicyModule",
    "ProspectiveEngine",
    "ProspectiveLog",
    "RewardModel",
    "RulePlanner",
    "SalienceEngine",
    "System2Plugin",
    "Task",
    "TaskQueue",
    "Telemetry",
    "TernaryField",
    "WorldState",
    # experiments
    "ExperimentConfig",
    "ExperimentResult",
    "compare_agents",
    "run_experiments",
    "run_single_config",
    "summarize_rows",
    # persistence
    "RunSnapshot",
    "latest_run",
    "list_runs",
    "load_snapshot",
    "save_snapshot",
]
