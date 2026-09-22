"""
app/ai_engine/reasoning/local_models.py
Local model capability discovery for Nirvexa AI Engine.
Inspects locally available models via the Ollama / local runtime without downloading,
pulling, or executing models.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
from urllib.parse import urlparse

import httpx
import openai

from config import Config

logger = logging.getLogger(__name__)

# Known visual encoder / multimodal architecture families
VISION_FAMILIES: Set[str] = {"clip", "mllama", "vision", "siglip", "blip"}

# Known text-only architecture families
TEXT_FAMILIES: Set[str] = {
    "llama",
    "mistral",
    "qwen",
    "qwen2",
    "gemma",
    "phi",
    "bert",
    "gpt",
    "deepseek",
}


@dataclass
class LocalModelInfo:
    """
    Metadata representation of an installed local model.
    """
    name: str
    available: bool = True
    capability: str = "unknown"  # "text", "vision", "unknown"
    size: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "capability": self.capability,
            "size": self.size,
            "details": dict(self.details),
        }


@dataclass
class ConfiguredModelStatus:
    """
    Resolution status of a configured local model against installed models.
    """
    model: str
    configured: bool
    available: bool
    capability: str = "unknown"
    reason: Optional[str] = None
    matched_installed_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "configured": self.configured,
            "available": self.available,
            "capability": self.capability,
            "reason": self.reason,
            "matched_installed_name": self.matched_installed_name,
        }


@dataclass
class LocalRuntimeStatus:
    """
    Aggregated health and capability status of the local runtime.
    """
    available: bool
    models: List[LocalModelInfo]
    configured_text_model: ConfiguredModelStatus
    configured_vision_model: ConfiguredModelStatus
    configured_text_model_available: bool
    configured_vision_model_available: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "models": [m.to_dict() for m in self.models],
            "configured_text_model": self.configured_text_model.to_dict(),
            "configured_vision_model": self.configured_vision_model.to_dict(),
            "configured_text_model_available": self.configured_text_model_available,
            "configured_vision_model_available": self.configured_vision_model_available,
            "metadata": dict(self.metadata),
        }


def _conf(key: str, default: Any) -> Any:
    """Safely retrieves configuration from Flask current_app or Config."""
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


def detect_capability_from_details(details: Optional[Dict[str, Any]]) -> str:
    """
    Determines model capability ('text', 'vision', 'unknown') from runtime metadata.
    Strictly avoids claiming vision unless reliable visual encoder metadata exists.
    """
    if not details or not isinstance(details, dict):
        return "unknown"

    raw_families = details.get("families") or []
    if isinstance(raw_families, str):
        raw_families = [raw_families]

    raw_family = details.get("family") or ""
    all_families = {str(f).strip().lower() for f in raw_families if f}
    if raw_family:
        all_families.add(str(raw_family).strip().lower())

    if not all_families:
        return "unknown"

    # Check for vision/clip architecture markers
    if any(any(vf in fam for vf in VISION_FAMILIES) for fam in all_families):
        return "vision"

    # Check for pure text architecture markers
    if any(any(tf in fam for tf in TEXT_FAMILIES) for fam in all_families):
        return "text"

    return "unknown"


def matches_model_name(configured: str, installed: str) -> bool:
    """
    Determines whether a configured model string matches an installed model tag.
    Examples:
        'llama3.2' matches 'llama3.2'
        'llama3.2' matches 'llama3.2:latest'
        'llama3.2:3b' matches 'llama3.2:3b'
        'llama3.2:3b' does not match 'llama3.2:latest'
    """
    cfg = configured.strip().lower()
    inst = installed.strip().lower()
    if not cfg or not inst:
        return False
    if cfg == inst:
        return True
    if cfg == inst.split(":")[0]:
        return True
    if f"{cfg}:latest" == inst:
        return True
    return False


class LocalModelDiscovery:
    """
    Read-only local model discovery and status resolver.
    Never executes model downloads, pulls, or destructive commands.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        raw_url = base_url or _conf(
            "AI_ENGINE_LOCAL_BASE_URL",
            _conf("AI_ENGINE_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
        self.base_url = str(raw_url).rstrip("/")
        self.api_key = str(api_key or _conf("AI_ENGINE_LOCAL_API_KEY", "ollama"))
        self.timeout_seconds = int(
            timeout_seconds
            if timeout_seconds is not None
            else _conf("AI_ENGINE_LOCAL_HEALTH_CHECK_TIMEOUT_SECONDS", 5)
        )

    def _get_root_url(self) -> str:
        """Strips trailing /v1 to access native Ollama API endpoints (/api/tags)."""
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    def _fetch_installed_models(
        self, timeout_seconds: Optional[int] = None
    ) -> Tuple[bool, List[LocalModelInfo]]:
        """
        Queries installed models from the local runtime without downloading anything.
        Tries native Ollama /api/tags first (for rich family/size metadata),
        then falls back to standard OpenAI /v1/models.
        """
        eff_timeout = float(
            timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        )
        root_url = self._get_root_url()
        tags_url = f"{root_url}/api/tags"

        # Attempt 1: Native Ollama /api/tags
        try:
            with httpx.Client(timeout=eff_timeout) as client:
                resp = client.get(tags_url)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_models = data.get("models") or []
                    model_list: List[LocalModelInfo] = []
                    for item in raw_models:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name") or item.get("model") or ""
                        if not name:
                            continue
                        details = item.get("details") or {}
                        cap = detect_capability_from_details(details)
                        size = item.get("size")
                        model_list.append(
                            LocalModelInfo(
                                name=name,
                                available=True,
                                capability=cap,
                                size=size if isinstance(size, int) else None,
                                details=details if isinstance(details, dict) else {},
                            )
                        )
                    return True, model_list
        except Exception as exc:
            logger.debug("[LocalModelDiscovery] Native /api/tags probe failed: %s", type(exc).__name__)

        # Attempt 2: OpenAI-compatible /v1/models endpoint
        try:
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=eff_timeout,
            )
            resp_models = client.models.list(timeout=eff_timeout)
            model_list = []
            for m in getattr(resp_models, "data", []):
                m_id = getattr(m, "id", None)
                if m_id:
                    model_list.append(
                        LocalModelInfo(
                            name=m_id,
                            available=True,
                            capability="unknown",
                            size=None,
                            details={},
                        )
                    )
            return True, model_list
        except Exception as exc:
            logger.warning(
                "[LocalModelDiscovery] Local runtime unavailable at %s: %s",
                self.base_url,
                type(exc).__name__,
            )
            return False, []

    def is_runtime_available(self, timeout_seconds: Optional[int] = None) -> bool:
        """Checks whether the local model runtime is online and responding."""
        available, _ = self._fetch_installed_models(timeout_seconds=timeout_seconds)
        return available

    def list_models(self, timeout_seconds: Optional[int] = None) -> List[LocalModelInfo]:
        """Returns the list of installed local models."""
        _, models = self._fetch_installed_models(timeout_seconds=timeout_seconds)
        return models

    def get_model(
        self, name: str, timeout_seconds: Optional[int] = None
    ) -> Optional[LocalModelInfo]:
        """Finds an installed model matching the given name or tag."""
        models = self.list_models(timeout_seconds=timeout_seconds)
        for m in models:
            if matches_model_name(name, m.name):
                return m
        return None

    def is_model_available(self, name: str, timeout_seconds: Optional[int] = None) -> bool:
        """Determines whether a specific model name is currently installed."""
        return self.get_model(name, timeout_seconds=timeout_seconds) is not None

    def get_text_model_status(
        self, timeout_seconds: Optional[int] = None
    ) -> ConfiguredModelStatus:
        """Evaluates whether the configured local text model is installed."""
        configured_model = str(
            _conf("AI_ENGINE_LOCAL_TEXT_MODEL", _conf("AI_ENGINE_LOCAL_MODEL", "llama3.2"))
        ).strip()
        available_runtime, models = self._fetch_installed_models(timeout_seconds=timeout_seconds)

        if not available_runtime:
            return ConfiguredModelStatus(
                model=configured_model,
                configured=bool(configured_model),
                available=False,
                capability="unknown",
                reason="runtime_unavailable",
            )

        for m in models:
            if matches_model_name(configured_model, m.name):
                return ConfiguredModelStatus(
                    model=configured_model,
                    configured=True,
                    available=True,
                    capability=m.capability,
                    reason="available",
                    matched_installed_name=m.name,
                )

        return ConfiguredModelStatus(
            model=configured_model,
            configured=True,
            available=False,
            capability="unknown",
            reason="model_not_installed",
        )

    def get_vision_model_status(
        self, timeout_seconds: Optional[int] = None
    ) -> ConfiguredModelStatus:
        """Evaluates whether the configured local vision model is configured and installed."""
        configured_model = str(_conf("AI_ENGINE_LOCAL_VISION_MODEL", "")).strip()

        is_configured = bool(
            configured_model and configured_model.lower() not in ("not_configured", "none")
        )

        if not is_configured:
            return ConfiguredModelStatus(
                model="",
                configured=False,
                available=False,
                capability="unknown",
                reason="vision_model_not_configured",
            )

        available_runtime, models = self._fetch_installed_models(timeout_seconds=timeout_seconds)

        if not available_runtime:
            return ConfiguredModelStatus(
                model=configured_model,
                configured=True,
                available=False,
                capability="unknown",
                reason="runtime_unavailable",
            )

        for m in models:
            if matches_model_name(configured_model, m.name):
                return ConfiguredModelStatus(
                    model=configured_model,
                    configured=True,
                    available=True,
                    capability=m.capability,
                    reason="available",
                    matched_installed_name=m.name,
                )

        return ConfiguredModelStatus(
            model=configured_model,
            configured=True,
            available=False,
            capability="unknown",
            reason="model_not_installed",
        )

    def get_runtime_status(
        self, timeout_seconds: Optional[int] = None
    ) -> LocalRuntimeStatus:
        """
        Returns a complete structured snapshot of local runtime health,
        installed models, and configured text/vision availability.
        """
        available_runtime, models = self._fetch_installed_models(timeout_seconds=timeout_seconds)
        text_status = self.get_text_model_status(timeout_seconds=timeout_seconds)
        vision_status = self.get_vision_model_status(timeout_seconds=timeout_seconds)

        return LocalRuntimeStatus(
            available=available_runtime,
            models=models,
            configured_text_model=text_status,
            configured_vision_model=vision_status,
            configured_text_model_available=text_status.available,
            configured_vision_model_available=vision_status.available,
            metadata={"base_url": self.base_url},
        )
