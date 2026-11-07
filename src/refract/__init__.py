"""Refract — a traceable LLM agent that turns a news event into a grounded, personalised insight.

One piece of news is “refracted” through the profile of each reader — a company, a sole proprietor, a
self-employed person or a private individual — into what it means for them: what happened, why it matters to
them, risks and opportunities, the institution's view and next steps. Every statement carries its origin (fact,
house view, AI inference, hypothesis) and its sources; code enforces grounding and caps the impact level.

    from refract import Refract

    rf = Refract.from_dir("examples/demo")
    run = rf.analyze("demo_importer", "ev_rate_hold_ru", role="cfo", language="en")
    print(run.insight.headline)
"""

from .agent import ABLATIONS, Agent, AgentConfig, AgentRun, ChatAgent, ablation
from .config import Settings, load_settings
from .facade import Refract
from .knowledge import DataStore
from .llm import LLMClient, ScriptedLLM

__version__ = "0.2.0"

__all__ = [
    "ABLATIONS",
    "Agent",
    "AgentConfig",
    "AgentRun",
    "ChatAgent",
    "DataStore",
    "LLMClient",
    "Refract",
    "ScriptedLLM",
    "Settings",
    "__version__",
    "ablation",
    "load_settings",
]
