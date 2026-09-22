"""
app/ai_engine/reasoning/model_policy.py
Resource-Aware Model Selection Policy for Nirvexa AI Engine (Phase 8).
Evaluates model suitability, resource constraints, and capability taxonomy
in a strictly read-only, advisory manner without executing, pulling,
downloading, or substituting models.
"""
import os
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

from config import Config
from app.ai_engine.reasoning.local_models import (
    LocalModelDiscovery,
    LocalModelInfo,
    VISION_FAMILIES,
    TEXT_FAMILIES,
)

logger = logging.getLogger(__name__)

# Capability Taxonomy
CAPABILITY_TEXT = "text"
CAPABILITY_VISION = "vision"
CAPABILITY_MULTIMODAL = "multimodal"
CAPABILITY_UNKNOWN = "unknown"

SUPPORTED_CAPABILITIES: Set[str] = {
    CAPABILITY_TEXT,
    CAPABILITY_VISION,
    CAPABILITY_MULTIMODAL,
    CAPABILITY_UNKNOWN,
}

# Standard Complexity Levels (Phase 8)
COMPLEXITY_LOW = "low"
COMPLEXITY_MEDIUM = "medium"
COMPLEXITY_HIGH = "high"

SUPPORTED_COMPLEXITIES: Set[str] = {
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    COMPLEXITY_HIGH,
}

# Standard Policy Reason Codes
CODE_CAPABILITY_MATCH = "CAPABILITY_MATCH"
CODE_CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
CODE_VISION_MODEL_REQUIRED = "VISION_MODEL_REQUIRED"
CODE_UNKNOWN_CAPABILITY = "UNKNOWN_CAPABILITY"
CODE_MODEL_NOT_INSTALLED = "MODEL_NOT_INSTALLED"
CODE_INSUFFICIENT_VRAM = "INSUFFICIENT_VRAM"
CODE_INSUFFICIENT_RAM = "INSUFFICIENT_RAM"
CODE_CONTEXT_TOO_SMALL = "CONTEXT_TOO_SMALL"
CODE_RESOURCE_REQUIREMENTS_UNKNOWN = "RESOURCE_REQUIREMENTS_UNKNOWN"
CODE_RESOURCE_CONSTRAINTS_SATISFIED = "RESOURCE_CONSTRAINTS_SATISFIED"
CODE_CLOUD_PREFERRED_TASK = "CLOUD_PREFERRED_TASK"
CODE_CONFIGURED_MODEL_SUITABLE = "CONFIGURED_MODEL_SUITABLE"
CODE_COMPLEXITY_MISMATCH = "COMPLEXITY_MISMATCH"

# Model Size Categories
SIZE_TINY = "tiny"        # <= 3B
SIZE_SMALL = "small"      # > 3B to 8B
SIZE_MEDIUM = "medium"    # > 8B to 20B
SIZE_LARGE = "large"      # > 20B
SIZE_UNKNOWN = "unknown"


def _conf(key: str, default: Any) -> Any:
    """Safely retrieves configuration from Flask current_app or Config."""
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


def parse_parameter_billions(details: Optional[Dict[str, Any]], model_name: str = "") -> Optional[float]:
    """
    Extracts parameter count in billions from model details or model tag.
    Returns None if unknown.
    """
    if details and isinstance(details, dict):
        param_size_str = str(details.get("parameter_size", "")).strip().upper()
        if param_size_str:
            # Handle forms like '3.2B', '7B', '70B', '0.5B', '500M'
            m_b = re.match(r"^([0-9.]+)\s*B$", param_size_str)
            if m_b:
                try:
                    return float(m_b.group(1))
                except ValueError:
                    pass
            m_m = re.match(r"^([0-9.]+)\s*M$", param_size_str)
            if m_m:
                try:
                    return round(float(m_m.group(1)) / 1000.0, 3)
                except ValueError:
                    pass

    # Heuristic extraction from model name if tag contains parameter indicator (e.g. ':3.2b', ':7b')
    if model_name:
        clean_name = str(model_name).lower()
        m_tag = re.search(r"[:_-]([0-9.]+)b\b", clean_name)
        if m_tag:
            try:
                return float(m_tag.group(1))
            except ValueError:
                pass

    return None


def classify_model_size(params_billions: Optional[float]) -> str:
    """Classifies model into tiny, small, medium, large, or unknown."""
    if params_billions is None or params_billions <= 0:
        return SIZE_UNKNOWN
    if params_billions <= 3.0:
        return SIZE_TINY
    if params_billions <= 8.0:
        return SIZE_SMALL
    if params_billions <= 20.0:
        return SIZE_MEDIUM
    return SIZE_LARGE


def parse_quantization(details: Optional[Dict[str, Any]], model_name: str = "") -> str:
    """Extracts quantization format (e.g. Q4, Q8, FP16) from details or name."""
    if details and isinstance(details, dict):
        raw_quant = str(details.get("quantization_level", "")).strip().upper()
        if raw_quant:
            return raw_quant

    if model_name:
        clean = str(model_name).upper()
        for q in ("Q4_K_M", "Q4_0", "Q4_1", "Q5_K_M", "Q5_0", "Q8_0", "FP16", "F16", "FP32", "F32", "Q4", "Q5", "Q6", "Q8", "Q2", "Q3"):
            if q in clean:
                return q

    return "unknown"


def estimate_vram_gb(params_b: Optional[float], quant: str) -> Optional[float]:
    """
    Estimates required GPU VRAM in GB based on parameters and quantization.
    Includes baseline runtime overhead. Returns None if parameters are unknown.
    """
    if params_b is None or params_b <= 0:
        return None

    q_upper = str(quant).upper()
    if "Q2" in q_upper:
        mult, overhead = 0.45, 0.8
    elif "Q3" in q_upper:
        mult, overhead = 0.55, 0.8
    elif "Q4" in q_upper:
        mult, overhead = 0.70, 1.0
    elif "Q5" in q_upper:
        mult, overhead = 0.85, 1.0
    elif "Q6" in q_upper:
        mult, overhead = 0.95, 1.0
    elif "Q8" in q_upper:
        mult, overhead = 1.15, 1.2
    elif "FP16" in q_upper or "F16" in q_upper:
        mult, overhead = 2.05, 1.5
    elif "FP32" in q_upper or "F32" in q_upper:
        mult, overhead = 4.10, 2.0
    else:
        # Default assumption: ~4-bit quant
        mult, overhead = 0.75, 1.0

    return round((params_b * mult) + overhead, 2)


def estimate_ram_gb(params_b: Optional[float], quant: str) -> Optional[float]:
    """
    Estimates required host RAM in GB if offloading or running on CPU.
    Returns None if parameters are unknown.
    """
    vram = estimate_vram_gb(params_b, quant)
    if vram is None:
        return None
    # CPU inference typically requires slight additional memory headroom
    return round(vram * 1.15, 2)


@dataclass
class LocalResourceProfile:
    """
    Lightweight, defensive hardware resource snapshot.
    Guarantees zero crashes if hardware metrics cannot be queried.
    Never executes network calls or sub-processes.
    """
    total_ram_gb: Optional[float] = None
    available_ram_gb: Optional[float] = None
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    cpu_cores: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "cpu_cores": self.cpu_cores,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def detect(cls) -> "LocalResourceProfile":
        """
        Defensively inspects system resources without external packages.
        """
        cores = os.cpu_count()
        total_ram = None
        avail_ram = None

        # Windows memory detection via ctypes
        if sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_ram = round(stat.ullTotalPhys / (1024 ** 3), 2)
                    avail_ram = round(stat.ullAvailPhys / (1024 ** 3), 2)
            except Exception as exc:
                logger.debug("[LocalResourceProfile] Windows RAM detection error: %s", exc)

        return cls(
            total_ram_gb=total_ram,
            available_ram_gb=avail_ram,
            gpu_available=False,
            gpu_name=None,
            gpu_vram_gb=None,
            cpu_cores=cores,
        )


@dataclass
class ModelRequirement:
    """
    Requirements specification for a Nirvexa task category.
    """
    task: str
    required_capability: str
    complexity: str
    preferred_target: str
    min_context_length: int = 2048
    max_recommended_size: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "required_capability": self.required_capability,
            "complexity": self.complexity,
            "preferred_target": self.preferred_target,
            "min_context_length": self.min_context_length,
            "max_recommended_size": self.max_recommended_size,
            "notes": self.notes,
        }


# Baseline Task Requirements
TASK_REQUIREMENTS: Dict[str, ModelRequirement] = {
    "resume": ModelRequirement(
        task="resume",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_LOW,
        preferred_target="local",
        min_context_length=2048,
        max_recommended_size=SIZE_SMALL,
        notes="Fast structured text processing suitable for SLMs",
    ),
    "classification": ModelRequirement(
        task="classification",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_LOW,
        preferred_target="local",
        min_context_length=1024,
        max_recommended_size=SIZE_SMALL,
        notes="Categorization and label assignment",
    ),
    "extraction": ModelRequirement(
        task="extraction",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_LOW,
        preferred_target="local",
        min_context_length=2048,
        max_recommended_size=SIZE_SMALL,
        notes="JSON/attribute extraction",
    ),
    "text_generation": ModelRequirement(
        task="text_generation",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_MEDIUM,
        preferred_target="cloud",
        min_context_length=4096,
        max_recommended_size=SIZE_MEDIUM,
    ),
    "chat": ModelRequirement(
        task="chat",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_MEDIUM,
        preferred_target="cloud",
        min_context_length=4096,
        max_recommended_size=SIZE_LARGE,
    ),
    "vision": ModelRequirement(
        task="vision",
        required_capability=CAPABILITY_VISION,
        complexity=COMPLEXITY_MEDIUM,
        preferred_target="vision_local",
        min_context_length=2048,
        max_recommended_size=SIZE_MEDIUM,
        notes="Visual inspection and OCR",
    ),
    "document_vision": ModelRequirement(
        task="document_vision",
        required_capability=CAPABILITY_VISION,
        complexity=COMPLEXITY_HIGH,
        preferred_target="vision_local",
        min_context_length=4096,
        max_recommended_size=SIZE_LARGE,
        notes="Multipage visual document understanding",
    ),
    "research": ModelRequirement(
        task="research",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_HIGH,
        preferred_target="cloud",
        min_context_length=8192,
        max_recommended_size=SIZE_LARGE,
        notes="Complex synthesis, web grounding, and citation validation",
    ),
    "salary_analysis": ModelRequirement(
        task="salary_analysis",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_HIGH,
        preferred_target="cloud",
        min_context_length=4096,
        max_recommended_size=SIZE_LARGE,
    ),
    "company_review": ModelRequirement(
        task="company_review",
        required_capability=CAPABILITY_TEXT,
        complexity=COMPLEXITY_HIGH,
        preferred_target="cloud",
        min_context_length=4096,
        max_recommended_size=SIZE_LARGE,
    ),
}


@dataclass
class ModelPolicyDecision:
    """
    Structured suitability evaluation returned by ModelPolicy.
    Pure advisory decision; never alters server-configured execution models.
    """
    suitable: bool
    model: str
    task: str
    capability: str
    reason_codes: List[str]
    size_category: str = SIZE_UNKNOWN
    estimated_vram_gb: Optional[float] = None
    estimated_ram_gb: Optional[float] = None
    context_length: Optional[int] = None
    recommended_fallback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suitable": self.suitable,
            "model": self.model,
            "task": self.task,
            "capability": self.capability,
            "reason_codes": list(self.reason_codes),
            "size_category": self.size_category,
            "estimated_vram_gb": self.estimated_vram_gb,
            "estimated_ram_gb": self.estimated_ram_gb,
            "context_length": self.context_length,
            "recommended_fallback": self.recommended_fallback,
            "metadata": dict(self.metadata),
        }


class ModelPolicy:
    """
    Resource-Aware Model Selection & Capability Evaluation Policy (Phase 8).
    Purely advisory decision layer.
    - Evaluates model suitability against task requirements and complexity levels.
    - Considers locally available models discovered by LocalModelDiscovery.
    - Enforces capability isolation (strict vision verification).
    - Checks hardware bounds (VRAM, RAM, context length) without crashing.
    - NEVER downloads, pulls, executes, or makes network requests.
    - NEVER silently substitutes configured models during execution.
    """

    def __init__(
        self,
        discovery: Optional[LocalModelDiscovery] = None,
        resource_profile: Optional[LocalResourceProfile] = None,
    ):
        self.discovery = discovery
        self.resource_profile = resource_profile or LocalResourceProfile.detect()

    def _resolve_available_models(
        self, available_models: Optional[List[LocalModelInfo]] = None
    ) -> List[LocalModelInfo]:
        """
        Resolves available models defensively without network calls if discovery is absent.
        """
        if available_models is not None:
            return available_models
        if self.discovery is not None:
            try:
                return self.discovery.list_models()
            except Exception:
                return []
        return []

    @classmethod
    def get_task_requirement(
        cls, task: str, complexity: Optional[str] = None
    ) -> ModelRequirement:
        """
        Resolves task requirement, adapting dynamically if an explicit complexity is requested.
        Handles unknown tasks and unknown complexities gracefully without crashing.
        """
        clean_task = str(task).strip().lower() if task else "text_generation"
        base_req = TASK_REQUIREMENTS.get(
            clean_task,
            ModelRequirement(
                task=clean_task,
                required_capability=CAPABILITY_TEXT,
                complexity=COMPLEXITY_MEDIUM,
                preferred_target="cloud",
                min_context_length=4096,
            ),
        )

        clean_complexity = (
            str(complexity).strip().lower() if complexity and isinstance(complexity, str) else None
        )

        # If complexity is unspecified or unknown, preserve default task requirement
        if not clean_complexity or clean_complexity not in SUPPORTED_COMPLEXITIES:
            return base_req

        # Specialize requirement based on requested complexity
        if clean_complexity == COMPLEXITY_LOW:
            return ModelRequirement(
                task=base_req.task,
                required_capability=base_req.required_capability,
                complexity=COMPLEXITY_LOW,
                preferred_target="local" if base_req.required_capability != CAPABILITY_VISION else "vision_local",
                min_context_length=min(base_req.min_context_length, 2048),
                max_recommended_size=SIZE_SMALL,
                notes="Optimized for low resource footprint and fast inference",
            )
        elif clean_complexity == COMPLEXITY_MEDIUM:
            return ModelRequirement(
                task=base_req.task,
                required_capability=base_req.required_capability,
                complexity=COMPLEXITY_MEDIUM,
                preferred_target=base_req.preferred_target,
                min_context_length=max(base_req.min_context_length, 4096),
                max_recommended_size=SIZE_MEDIUM,
                notes="Balanced capability and resource consumption",
            )
        else:  # COMPLEXITY_HIGH
            return ModelRequirement(
                task=base_req.task,
                required_capability=base_req.required_capability,
                complexity=COMPLEXITY_HIGH,
                preferred_target="cloud" if base_req.required_capability != CAPABILITY_VISION else "vision_local",
                min_context_length=max(base_req.min_context_length, 4096),
                max_recommended_size=SIZE_LARGE,
                notes="Requires high reasoning capacity or cloud offload",
            )

    def evaluate_suitability(
        self,
        task: str,
        model_name: str,
        complexity: Optional[str] = None,
        available_models: Optional[List[LocalModelInfo]] = None,
    ) -> ModelPolicyDecision:
        """
        Evaluates whether a model is suitable for a specific task and complexity.
        Checks capability matching, model installation, and resource limits.
        Pure, read-only evaluation.
        """
        clean_task = str(task).strip().lower() if task else "text_generation"
        target_model = str(model_name).strip() if model_name else ""
        requirement = self.get_task_requirement(clean_task, complexity=complexity)

        reason_codes: List[str] = []
        is_suitable = True

        # 1. Resolve model info from discovery or available_models
        models = self._resolve_available_models(available_models)
        matched_info: Optional[LocalModelInfo] = None
        if target_model:
            for m in models:
                if (
                    m.name == target_model
                    or m.name.startswith(f"{target_model}:")
                    or target_model.startswith(f"{m.name}:")
                ):
                    matched_info = m
                    break

        if not matched_info:
            is_suitable = False
            reason_codes.append(CODE_MODEL_NOT_INSTALLED)
            capability = CAPABILITY_UNKNOWN
            params_b = parse_parameter_billions({}, target_model)
            quant = parse_quantization({}, target_model)
            context_len = _conf("AI_ENGINE_LOCAL_DEFAULT_CONTEXT_LENGTH", 4096)
        else:
            capability = matched_info.capability
            params_b = parse_parameter_billions(matched_info.details, matched_info.name)
            quant = parse_quantization(matched_info.details, matched_info.name)
            context_len = matched_info.details.get("context_length") or _conf(
                "AI_ENGINE_LOCAL_DEFAULT_CONTEXT_LENGTH", 4096
            )

        size_category = classify_model_size(params_b)
        est_vram = estimate_vram_gb(params_b, quant)
        est_ram = estimate_ram_gb(params_b, quant)

        # 2. Capability Compatibility Verification
        req_cap = requirement.required_capability

        if req_cap == CAPABILITY_VISION:
            # Strict Vision Safety: Text models and unverified models are strictly rejected
            if capability in (CAPABILITY_VISION, CAPABILITY_MULTIMODAL):
                reason_codes.append(CODE_CAPABILITY_MATCH)
            elif capability == CAPABILITY_UNKNOWN:
                require_known = _conf("AI_ENGINE_LOCAL_REQUIRE_KNOWN_VISION_CAPABILITY", True)
                if require_known:
                    is_suitable = False
                    reason_codes.append(CODE_UNKNOWN_CAPABILITY)
                    reason_codes.append(CODE_VISION_MODEL_REQUIRED)
                else:
                    reason_codes.append(CODE_CAPABILITY_MATCH)
            else:
                is_suitable = False
                reason_codes.append(CODE_CAPABILITY_MISMATCH)
                reason_codes.append(CODE_VISION_MODEL_REQUIRED)

        elif req_cap == CAPABILITY_TEXT:
            if capability in (CAPABILITY_TEXT, CAPABILITY_MULTIMODAL, CAPABILITY_UNKNOWN):
                reason_codes.append(CODE_CAPABILITY_MATCH)
            else:
                is_suitable = False
                reason_codes.append(CODE_CAPABILITY_MISMATCH)

        # 3. Context Length Validation
        if context_len < requirement.min_context_length:
            is_suitable = False
            reason_codes.append(CODE_CONTEXT_TOO_SMALL)

        # 4. Resource Constraint Evaluation
        max_vram_cfg = float(_conf("AI_ENGINE_LOCAL_MAX_MODEL_VRAM_GB", 0.0) or 0.0)
        max_ram_cfg = float(_conf("AI_ENGINE_LOCAL_MAX_MODEL_RAM_GB", 0.0) or 0.0)

        if est_vram is not None:
            if max_vram_cfg > 0.0 and est_vram > max_vram_cfg:
                is_suitable = False
                reason_codes.append(CODE_INSUFFICIENT_VRAM)
            elif self.resource_profile.gpu_vram_gb is not None and est_vram > self.resource_profile.gpu_vram_gb:
                is_suitable = False
                reason_codes.append(CODE_INSUFFICIENT_VRAM)

        if est_ram is not None:
            if max_ram_cfg > 0.0 and est_ram > max_ram_cfg:
                is_suitable = False
                reason_codes.append(CODE_INSUFFICIENT_RAM)
            elif self.resource_profile.total_ram_gb is not None and est_ram > self.resource_profile.total_ram_gb:
                is_suitable = False
                reason_codes.append(CODE_INSUFFICIENT_RAM)
            elif (
                self.resource_profile.total_ram_gb is None
                and self.resource_profile.available_ram_gb is not None
                and est_ram > self.resource_profile.available_ram_gb
            ):
                is_suitable = False
                reason_codes.append(CODE_INSUFFICIENT_RAM)

        if est_vram is None and est_ram is None:
            reason_codes.append(CODE_RESOURCE_REQUIREMENTS_UNKNOWN)
        elif is_suitable and CODE_INSUFFICIENT_VRAM not in reason_codes and CODE_INSUFFICIENT_RAM not in reason_codes:
            reason_codes.append(CODE_RESOURCE_CONSTRAINTS_SATISFIED)

        # 5. Cloud-Preferred Advisory Flag
        if requirement.preferred_target == "cloud":
            reason_codes.append(CODE_CLOUD_PREFERRED_TASK)

        # 6. Complexity Guidance Check
        if requirement.complexity == COMPLEXITY_HIGH and size_category == SIZE_TINY:
            reason_codes.append(CODE_COMPLEXITY_MISMATCH)

        return ModelPolicyDecision(
            suitable=is_suitable,
            model=target_model,
            task=clean_task,
            capability=capability,
            reason_codes=reason_codes,
            size_category=size_category,
            estimated_vram_gb=est_vram,
            estimated_ram_gb=est_ram,
            context_length=context_len,
            recommended_fallback="cloud" if not is_suitable else None,
            metadata={
                "quantization": quant,
                "parameter_count_billions": params_b,
                "complexity": requirement.complexity,
            },
        )

    def recommend_model(
        self,
        task: str,
        complexity: Optional[str] = None,
        available_models: Optional[List[LocalModelInfo]] = None,
    ) -> Optional[str]:
        """
        Advisory recommendation engine.
        Finds the most suitable installed local model for a task and requested complexity.
        - LOW: prefers smallest suitable model (<= 3B preferred, up to 8B acceptable).
        - MEDIUM: prefers balanced models (3B to 14B).
        - HIGH: prefers largest suitable local model (>= 7B, 14B+, 20B+ where available).
        NEVER downloads models and NEVER silently replaces configured models during execution.
        """
        clean_task = str(task).strip().lower() if task else "text_generation"
        requirement = self.get_task_requirement(clean_task, complexity=complexity)
        models = self._resolve_available_models(available_models)

        if not models:
            return None

        # Resolve configured model
        if requirement.required_capability == CAPABILITY_VISION:
            configured_model = str(_conf("AI_ENGINE_LOCAL_VISION_MODEL", "")).strip()
        else:
            configured_model = str(
                _conf("AI_ENGINE_LOCAL_TEXT_MODEL", _conf("AI_ENGINE_LOCAL_MODEL", "llama3.2"))
            ).strip()

        # Check if configured model is installed and suitable
        decision_configured = (
            self.evaluate_suitability(clean_task, configured_model, complexity=complexity, available_models=models)
            if configured_model
            else None
        )

        # Collect suitable installed models
        suitable_candidates: List[Tuple[float, LocalModelInfo]] = []
        for m in models:
            dec = self.evaluate_suitability(clean_task, m.name, complexity=complexity, available_models=models)
            if dec.suitable:
                params_b = parse_parameter_billions(m.details, m.name)
                score = params_b if params_b is not None else 7.0
                suitable_candidates.append((score, m))

        if not suitable_candidates:
            return None

        req_complexity = requirement.complexity

        # Rule 1: Complexity LOW
        # Prefer smallest suitable model (<= 3B preferred, up to 8B)
        if req_complexity == COMPLEXITY_LOW:
            # Sort ascending by parameter count
            suitable_candidates.sort(key=lambda x: (x[0], x[1].name))
            # If configured model is suitable and in the low tier (<= 8B), prefer configured
            if decision_configured and decision_configured.suitable:
                conf_b = parse_parameter_billions({}, configured_model) or 3.2
                if conf_b <= 8.0 and complexity is None:
                    return configured_model
            return suitable_candidates[0][1].name

        # Rule 2: Complexity MEDIUM
        # Prefer balanced models (3B to 14B)
        elif req_complexity == COMPLEXITY_MEDIUM:
            # Separate models in balanced tier [3B, 14B] from outliers
            balanced = [c for c in suitable_candidates if 3.0 <= c[0] <= 14.0]
            if balanced:
                # Prefer balanced candidate closest to standard ~7B-8B
                balanced.sort(key=lambda x: (abs(x[0] - 7.5), x[1].name))
                if decision_configured and decision_configured.suitable:
                    conf_b = parse_parameter_billions({}, configured_model) or 7.0
                    if 3.0 <= conf_b <= 14.0 and complexity is None:
                        return configured_model
                return balanced[0][1].name
            else:
                # Fallback to closest available suitable candidate
                suitable_candidates.sort(key=lambda x: (abs(x[0] - 7.5), x[1].name))
                return suitable_candidates[0][1].name

        # Rule 3: Complexity HIGH
        # Prefer largest suitable local model (>= 7B, 14B+, 20B+)
        else:
            # Sort descending by parameter count
            suitable_candidates.sort(key=lambda x: (-x[0], x[1].name))
            # If configured model is suitable and large (>= 7B), prefer it if complexity was not overridden
            if decision_configured and decision_configured.suitable:
                conf_b = parse_parameter_billions({}, configured_model) or 0.0
                if conf_b >= 7.0 and complexity is None:
                    return configured_model
            return suitable_candidates[0][1].name

    def select_model(
        self,
        task: str,
        complexity: Optional[str] = None,
        available_models: Optional[List[LocalModelInfo]] = None,
    ) -> ModelPolicyDecision:
        """
        Pure advisory model selection evaluation.
        Evaluates task and complexity against installed models and returns a deterministic
        ModelPolicyDecision. Never alters server-configured execution models.
        """
        clean_task = str(task).strip().lower() if task else "text_generation"
        models = self._resolve_available_models(available_models)
        rec_name = self.recommend_model(clean_task, complexity=complexity, available_models=models)

        if rec_name:
            decision = self.evaluate_suitability(
                clean_task, rec_name, complexity=complexity, available_models=models
            )
            decision.metadata["recommended_model"] = rec_name
            decision.metadata["selection_mode"] = "local_recommendation"
            return decision

        # No suitable local model found
        req = self.get_task_requirement(clean_task, complexity=complexity)
        return ModelPolicyDecision(
            suitable=False,
            model="",
            task=clean_task,
            capability=req.required_capability,
            reason_codes=[CODE_MODEL_NOT_INSTALLED, CODE_CLOUD_PREFERRED_TASK],
            recommended_fallback="cloud",
            metadata={
                "complexity": req.complexity,
                "selection_mode": "cloud_fallback_recommended",
            },
        )


# Backward-compatible alias
LocalModelPolicy = ModelPolicy

__all__ = [
    "ModelPolicy",
    "LocalModelPolicy",
    "ModelPolicyDecision",
    "ModelRequirement",
    "LocalResourceProfile",
    "COMPLEXITY_LOW",
    "COMPLEXITY_MEDIUM",
    "COMPLEXITY_HIGH",
    "SUPPORTED_COMPLEXITIES",
    "CAPABILITY_TEXT",
    "CAPABILITY_VISION",
    "CAPABILITY_MULTIMODAL",
    "CAPABILITY_UNKNOWN",
    "CODE_CAPABILITY_MATCH",
    "CODE_CAPABILITY_MISMATCH",
    "CODE_VISION_MODEL_REQUIRED",
    "CODE_UNKNOWN_CAPABILITY",
    "CODE_MODEL_NOT_INSTALLED",
    "CODE_INSUFFICIENT_VRAM",
    "CODE_INSUFFICIENT_RAM",
    "CODE_CONTEXT_TOO_SMALL",
    "CODE_RESOURCE_REQUIREMENTS_UNKNOWN",
    "CODE_RESOURCE_CONSTRAINTS_SATISFIED",
    "CODE_CLOUD_PREFERRED_TASK",
    "CODE_CONFIGURED_MODEL_SUITABLE",
    "CODE_COMPLEXITY_MISMATCH",
    "SIZE_TINY",
    "SIZE_SMALL",
    "SIZE_MEDIUM",
    "SIZE_LARGE",
    "SIZE_UNKNOWN",
    "parse_parameter_billions",
    "classify_model_size",
    "parse_quantization",
    "estimate_vram_gb",
    "estimate_ram_gb",
]
