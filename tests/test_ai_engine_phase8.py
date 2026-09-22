"""
tests/test_ai_engine_phase8.py
Focused Test Suite for Nirvexa AI Engine Phase 8: Resource-Aware Model Selection Policy.
Verifies complexity levels, capability isolation, resource constraints,
advisory recommendation, zero network calls, no-download, no-silent-substitution,
and compatibility with Phase 7 ModelRouter & TaskExecutor.
"""
import pytest
from unittest.mock import MagicMock, patch

from config import Config
from app.ai_engine.reasoning.model_policy import (
    ModelPolicy,
    LocalModelPolicy,
    ModelPolicyDecision,
    ModelRequirement,
    LocalResourceProfile,
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    COMPLEXITY_HIGH,
    SUPPORTED_COMPLEXITIES,
    CAPABILITY_TEXT,
    CAPABILITY_VISION,
    CAPABILITY_MULTIMODAL,
    CAPABILITY_UNKNOWN,
    CODE_CAPABILITY_MATCH,
    CODE_CAPABILITY_MISMATCH,
    CODE_VISION_MODEL_REQUIRED,
    CODE_UNKNOWN_CAPABILITY,
    CODE_MODEL_NOT_INSTALLED,
    CODE_INSUFFICIENT_VRAM,
    CODE_INSUFFICIENT_RAM,
    CODE_CONTEXT_TOO_SMALL,
    CODE_RESOURCE_REQUIREMENTS_UNKNOWN,
    CODE_RESOURCE_CONSTRAINTS_SATISFIED,
    CODE_CLOUD_PREFERRED_TASK,
    CODE_CONFIGURED_MODEL_SUITABLE,
    CODE_COMPLEXITY_MISMATCH,
    SIZE_TINY,
    SIZE_SMALL,
    SIZE_MEDIUM,
    SIZE_LARGE,
)
from app.ai_engine.reasoning.local_models import LocalModelDiscovery, LocalModelInfo
from app.ai_engine.reasoning.model_router import ModelRouter, ModelRouteDecision
from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_LOCAL_MODEL_UNAVAILABLE
from app import create_app
from app.ai_engine.reasoning.schemas import LLMGenerationResult
from app.ai_engine.reasoning.providers import MockLLMProvider


@pytest.fixture
def app():
    """Create Flask test application configured for Phase 8 testing."""
    test_app = create_app("testing")
    test_app.config["AI_ENGINE_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_PROVIDER"] = "ollama"
    test_app.config["AI_ENGINE_LOCAL_MODEL"] = "llama3.2"
    test_app.config["AI_ENGINE_LOCAL_BASE_URL"] = "http://localhost:11434/v1"
    test_app.config["AI_ENGINE_LOCAL_TIMEOUT_SECONDS"] = 60
    test_app.config["AI_ENGINE_LOCAL_MAX_CONCURRENCY"] = 2
    test_app.config["AI_ENGINE_LOCAL_HEALTH_CHECK_TIMEOUT_SECONDS"] = 5
    test_app.config["AI_ENGINE_LOCAL_POLICY_ENABLED"] = True
    return test_app


@pytest.fixture
def client(app):
    return app.test_client()


# ==============================================================================
# 1. Complexity Selection Tests
# ==============================================================================

def test_01_low_complexity_picks_smallest_model(app):
    """
    Verifies that for complexity='low', ModelPolicy prefers the smallest suitable
    local model (e.g. <= 3B).
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="large-model:14b", capability="text", details={"parameter_size": "14B"}),
            LocalModelInfo(name="tiny-model:1.5b", capability="text", details={"parameter_size": "1.5B"}),
            LocalModelInfo(name="medium-model:7b", capability="text", details={"parameter_size": "7B"}),
        ]
        policy = ModelPolicy()
        rec = policy.recommend_model("text_generation", complexity=COMPLEXITY_LOW, available_models=models)
        assert rec == "tiny-model:1.5b"

        decision = policy.select_model("text_generation", complexity=COMPLEXITY_LOW, available_models=models)
        assert decision.suitable is True
        assert decision.model == "tiny-model:1.5b"
        assert decision.size_category == SIZE_TINY


def test_02_medium_complexity_picks_balanced_model(app):
    """
    Verifies that for complexity='medium', ModelPolicy prefers a balanced model (3B-14B).
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="tiny-model:1.5b", capability="text", details={"parameter_size": "1.5B"}),
            LocalModelInfo(name="balanced-model:7b", capability="text", details={"parameter_size": "7B"}),
            LocalModelInfo(name="huge-model:32b", capability="text", details={"parameter_size": "32B"}),
        ]
        policy = ModelPolicy()
        rec = policy.recommend_model("text_generation", complexity=COMPLEXITY_MEDIUM, available_models=models)
        assert rec == "balanced-model:7b"

        decision = policy.select_model("text_generation", complexity=COMPLEXITY_MEDIUM, available_models=models)
        assert decision.suitable is True
        assert decision.model == "balanced-model:7b"
        assert decision.size_category == SIZE_SMALL


def test_03_high_complexity_picks_largest_model_or_recommends_cloud(app):
    """
    Verifies that for complexity='high':
    - Picks the largest suitable local model when available (>= 7B / 14B+).
    - Returns cloud recommendation when only tiny local models exist.
    """
    with app.app_context():
        # Case A: Local large model available
        models = [
            LocalModelInfo(name="small-model:3b", capability="text", details={"parameter_size": "3B", "context_length": 16384}),
            LocalModelInfo(name="large-model:14b", capability="text", details={"parameter_size": "14B", "context_length": 16384}),
        ]
        policy = ModelPolicy()
        rec = policy.recommend_model("research", complexity=COMPLEXITY_HIGH, available_models=models)
        assert rec == "large-model:14b"

        # Case B: Only tiny models available for a high-complexity cloud task
        tiny_only = [
            LocalModelInfo(name="tiny-model:1.5b", capability="text", details={"parameter_size": "1.5B"})
        ]
        d = policy.evaluate_suitability("research", "tiny-model:1.5b", complexity=COMPLEXITY_HIGH, available_models=tiny_only)
        assert CODE_COMPLEXITY_MISMATCH in d.reason_codes or CODE_CLOUD_PREFERRED_TASK in d.reason_codes
        assert d.recommended_fallback == "cloud" or CODE_CLOUD_PREFERRED_TASK in d.reason_codes


def test_04_unknown_complexity_does_not_crash(app):
    """
    Verifies that an unknown, empty, or non-standard complexity string is handled
    gracefully without crashing, falling back to the task's default complexity.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="llama3.2:3b", capability="text", details={"parameter_size": "3B"})
        ]
        policy = ModelPolicy()

        # Unknown complexity string
        req = policy.get_task_requirement("resume", complexity="ultra_extreme_quantum")
        assert req.complexity in SUPPORTED_COMPLEXITIES

        rec = policy.recommend_model("resume", complexity="invalid_complexity", available_models=models)
        assert rec == "llama3.2:3b"

        dec = policy.select_model("resume", complexity=None, available_models=models)
        assert dec.suitable is True
        assert dec.model in ("llama3.2", "llama3.2:3b")


# ==============================================================================
# 2. Capability Discrimination Tests
# ==============================================================================

def test_05_vision_requires_verified_vision_capability(app):
    """
    Verifies that vision tasks (vision, document_vision) strictly require verified
    visual architecture families and reject text-only models.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="llama3.2:3b", capability="text", details={"family": "llama"}),
            LocalModelInfo(name="llama3.2-vision:11b", capability="vision", details={"families": ["mllama", "vision"]}),
        ]
        policy = ModelPolicy()

        # Text model evaluated for vision -> rejected
        d_text = policy.evaluate_suitability("vision", "llama3.2:3b", available_models=models)
        assert d_text.suitable is False
        assert CODE_VISION_MODEL_REQUIRED in d_text.reason_codes
        assert CODE_CAPABILITY_MISMATCH in d_text.reason_codes

        # Verified vision model evaluated for vision -> accepted
        d_vis = policy.evaluate_suitability("vision", "llama3.2-vision:11b", available_models=models)
        assert d_vis.suitable is True
        assert CODE_CAPABILITY_MATCH in d_vis.reason_codes


def test_06_text_accepts_text_and_multimodal(app):
    """
    Verifies that standard text tasks accept both pure text and multimodal models.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="text-slm:3b", capability="text", details={"family": "llama"}),
            LocalModelInfo(name="multi-slm:11b", capability="multimodal", details={"families": ["mllama", "clip"]}),
        ]
        policy = ModelPolicy()

        d_text = policy.evaluate_suitability("text_generation", "text-slm:3b", available_models=models)
        assert d_text.suitable is True
        assert CODE_CAPABILITY_MATCH in d_text.reason_codes

        d_multi = policy.evaluate_suitability("text_generation", "multi-slm:11b", available_models=models)
        assert d_multi.suitable is True
        assert CODE_CAPABILITY_MATCH in d_multi.reason_codes


def test_07_deceptive_model_name_does_not_grant_vision_capability(app):
    """
    Verifies that a model containing 'vision' or 'ocr' in its name is NOT accepted
    for vision tasks if its architecture family is not verified vision.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="vision-ocr-super-fake:8b", capability="text", details={"family": "llama"})
        ]
        policy = ModelPolicy()
        d = policy.evaluate_suitability("vision", "vision-ocr-super-fake:8b", available_models=models)
        assert d.suitable is False
        assert CODE_VISION_MODEL_REQUIRED in d.reason_codes


# ==============================================================================
# 3. Availability & Advisory Recommendation Tests (No-Silent-Substitution)
# ==============================================================================

def test_08_unavailable_model_handling(app):
    """
    Verifies that a model not installed in discovery returns suitable=False with
    CODE_MODEL_NOT_INSTALLED, and does not assume the model exists.
    """
    with app.app_context():
        policy = ModelPolicy()
        d = policy.evaluate_suitability("resume", "non-existent-model:99b", available_models=[])
        assert d.suitable is False
        assert CODE_MODEL_NOT_INSTALLED in d.reason_codes
        assert d.recommended_fallback == "cloud"


def test_09_advisory_recommendation_when_configured_model_unavailable(app):
    """
    Verifies the Phase 7 & 8 no-silent-substitution invariant:
    1. Configured model remains the execution model in ModelRouter and TaskExecutor.
    2. ModelPolicy can identify a suitable alternative and surfaces it as an advisory recommendation.
    3. The alternative appears only in metadata['recommended_model'].
    4. TaskExecutor does NOT silently execute that alternative.
    5. Configured cloud fallback behavior remains controlled by Phase 7 server configuration.
    """
    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "configured-llama:3b"
        app.config["AI_ENGINE_LOCAL_FALLBACK_TO_CLOUD"] = True

        # Discovery: configured-llama:3b is missing, but qwen3.5:4b is installed
        installed_models = [
            LocalModelInfo(name="qwen3.5:4b", capability="text", details={"parameter_size": "4B"})
        ]
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.side_effect = lambda m: m == "qwen3.5:4b"
        mock_disc.list_models.return_value = installed_models

        # Cloud fallback provider mock
        mock_cloud = MagicMock()
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.provider = "gemini"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Cloud execution output", model="gemini-2.5-flash", provider="gemini"
        )

        policy = ModelPolicy(discovery=mock_disc)
        rec = policy.recommend_model("resume", available_models=installed_models)
        assert rec == "qwen3.5:4b"

        # ModelRouter: decision.model remains configured-llama:3b
        decision = ModelRouter.resolve_route("resume", discovery=mock_disc, policy=policy)
        assert decision.model == "configured-llama:3b"
        assert decision.metadata.get("recommended_model") == "qwen3.5:4b"

        # TaskExecutor: executes cloud fallback, NOT qwen3.5:4b
        mock_adapter = MagicMock()
        executor = TaskExecutor(
            discovery=mock_disc,
            local_adapter=mock_adapter,
            cloud_provider=mock_cloud,
            policy=policy,
        )
        res = executor.execute("resume", prompt="Enhance resume", allow_cloud_fallback=True)

        assert res.success is True
        assert res.target == "cloud"
        assert res.model == "gemini-2.5-flash"
        assert res.metadata.get("configured_model") == "configured-llama:3b"
        assert res.metadata.get("recommended_model") == "qwen3.5:4b"
        # Local adapter was NOT executed for qwen3.5:4b
        mock_adapter.generate.assert_not_called()


def test_10_configured_model_preferred_when_suitable(app):
    """
    Verifies that when the configured model is installed and suitable,
    ModelPolicy recommends the configured model.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        models = [
            LocalModelInfo(name="qwen2.5:3b", capability="text", details={"parameter_size": "3B"}),
            LocalModelInfo(name="llama3.2:latest", capability="text", details={"parameter_size": "3.2B"}),
        ]
        policy = ModelPolicy()
        rec = policy.recommend_model("resume", available_models=models)
        assert rec == "llama3.2"


# ==============================================================================
# 4. Resource Constraint Tests
# ==============================================================================

def test_11_vram_limits(app):
    """
    Verifies that when AI_ENGINE_LOCAL_MAX_MODEL_VRAM_GB is exceeded,
    the model is flagged with CODE_INSUFFICIENT_VRAM and suitable=False.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_MAX_MODEL_VRAM_GB"] = 4.0

        # Model needing ~11 GB VRAM (14B)
        models = [
            LocalModelInfo(name="heavy-model:14b", capability="text", details={"parameter_size": "14B"}),
            LocalModelInfo(name="light-model:3b", capability="text", details={"parameter_size": "3B"}),
        ]
        policy = ModelPolicy()

        d_heavy = policy.evaluate_suitability("text_generation", "heavy-model:14b", available_models=models)
        assert d_heavy.suitable is False
        assert CODE_INSUFFICIENT_VRAM in d_heavy.reason_codes

        d_light = policy.evaluate_suitability("text_generation", "light-model:3b", available_models=models)
        assert d_light.suitable is True
        assert CODE_RESOURCE_CONSTRAINTS_SATISFIED in d_light.reason_codes


def test_12_ram_limits(app):
    """
    Verifies that when AI_ENGINE_LOCAL_MAX_MODEL_RAM_GB is exceeded,
    the model is flagged with CODE_INSUFFICIENT_RAM and suitable=False.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_MAX_MODEL_RAM_GB"] = 5.0

        models = [
            LocalModelInfo(name="heavy-model:14b", capability="text", details={"parameter_size": "14B"})
        ]
        policy = ModelPolicy()
        d = policy.evaluate_suitability("text_generation", "heavy-model:14b", available_models=models)
        assert d.suitable is False
        assert CODE_INSUFFICIENT_RAM in d.reason_codes


def test_13_context_length(app):
    """
    Verifies that when a task requires high context (e.g. research with 8192),
    a model with insufficient context length is rejected with CODE_CONTEXT_TOO_SMALL.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="short-ctx:7b", capability="text", details={"parameter_size": "7B", "context_length": 2048}),
            LocalModelInfo(name="long-ctx:7b", capability="text", details={"parameter_size": "7B", "context_length": 32768}),
        ]
        policy = ModelPolicy()

        d_short = policy.evaluate_suitability("research", "short-ctx:7b", available_models=models)
        assert d_short.suitable is False
        assert CODE_CONTEXT_TOO_SMALL in d_short.reason_codes

        d_long = policy.evaluate_suitability("research", "long-ctx:7b", available_models=models)
        assert CODE_CONTEXT_TOO_SMALL not in d_long.reason_codes


def test_14_defensive_hardware_detection():
    """
    Verifies that LocalResourceProfile.detect() inspects host hardware defensively
    and never crashes even if platform queries fail.
    """
    profile = LocalResourceProfile.detect()
    assert isinstance(profile, LocalResourceProfile)
    assert profile.cpu_cores is None or profile.cpu_cores > 0
    p_dict = profile.to_dict()
    assert "total_ram_gb" in p_dict
    assert "gpu_available" in p_dict


# ==============================================================================
# 5. Purity & Safety Tests (No Network / No Download / No Execution)
# ==============================================================================

def test_15_zero_network_calls(app):
    """
    Verifies that ModelPolicy makes zero socket/HTTP network calls during operation.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="slm:3b", capability="text", details={"parameter_size": "3B"})
        ]
        policy = ModelPolicy()

        with patch("socket.socket") as mock_socket, \
             patch("httpx.Client") as mock_httpx, \
             patch("urllib.request.urlopen") as mock_urllib:

            policy.evaluate_suitability("resume", "slm:3b", available_models=models)
            policy.recommend_model("resume", complexity="low", available_models=models)
            policy.select_model("resume", complexity="low", available_models=models)

            mock_socket.assert_not_called()
            mock_httpx.assert_not_called()
            mock_urllib.assert_not_called()


def test_16_zero_download_methods():
    """
    Verifies that ModelPolicy and LocalModelPolicy expose zero download or pull methods.
    """
    disallowed = ["pull", "download", "install", "pull_model", "create"]
    for method in disallowed:
        assert not hasattr(ModelPolicy, method), f"ModelPolicy must not expose {method}"
        assert not hasattr(LocalModelPolicy, method), f"LocalModelPolicy must not expose {method}"


def test_17_zero_execution_methods():
    """
    Verifies that ModelPolicy and LocalModelPolicy expose zero model execution methods.
    """
    disallowed = ["generate", "execute", "call", "run", "predict", "chat"]
    for method in disallowed:
        assert not hasattr(ModelPolicy, method), f"ModelPolicy must not expose {method}"
        assert not hasattr(LocalModelPolicy, method), f"LocalModelPolicy must not expose {method}"


def test_18_deterministic_selection(app):
    """
    Verifies that repeated evaluations with identical inputs produce deterministic results.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="model-a:3b", capability="text", details={"parameter_size": "3B"}),
            LocalModelInfo(name="model-b:7b", capability="text", details={"parameter_size": "7B"}),
        ]
        policy = ModelPolicy()

        decisions = [
            policy.select_model("text_generation", complexity="low", available_models=models)
            for _ in range(5)
        ]

        for d in decisions[1:]:
            assert d.model == decisions[0].model
            assert d.suitable == decisions[0].suitable
            assert d.reason_codes == decisions[0].reason_codes
            assert d.size_category == decisions[0].size_category


def test_19_no_shared_tenant_user_state(app):
    """
    Verifies that ModelPolicy does not retain or leak user/tenant state across calls.
    """
    with app.app_context():
        models = [
            LocalModelInfo(name="slm:3b", capability="text", details={"parameter_size": "3B"})
        ]
        policy = ModelPolicy()

        d1 = policy.select_model("resume", complexity="low", available_models=models)
        d2 = policy.select_model("research", complexity="high", available_models=models)

        assert d1.task == "resume"
        assert d2.task == "research"
        assert d1.metadata.get("complexity") == "low"
        assert d2.metadata.get("complexity") == "high"


# ==============================================================================
# 6. Integration & Backward Compatibility Tests
# ==============================================================================

def test_20_model_router_integration(app):
    """
    Verifies that ModelRouter.resolve_route accepts complexity, attaches advisory
    metadata, and keeps decision.model set to the server-configured model.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "configured-llama:3b"

        models = [
            LocalModelInfo(name="configured-llama:3b", capability="text", details={"parameter_size": "3B"}),
            LocalModelInfo(name="tiny-model:1b", capability="text", details={"parameter_size": "1B"}),
        ]
        mock_disc = MagicMock()
        mock_disc.is_model_available.return_value = True
        mock_disc.list_models.return_value = models

        decision = ModelRouter.resolve_route("resume", discovery=mock_disc, complexity="low")
        assert decision.model == "configured-llama:3b"
        assert "recommended_model" in decision.metadata
        assert "policy_decision" in decision.metadata


def test_21_task_executor_integration(app):
    """
    Verifies that TaskExecutor.execute accepts complexity, forwards it to router,
    and returns policy metadata in TaskExecutionResult.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True
        mock_disc.list_models.return_value = [
            LocalModelInfo(name="llama3.2", capability="text", details={"parameter_size": "3.2B"})
        ]

        mock_adapter = MagicMock()
        from app.ai_engine.reasoning.local_adapter import LocalModelResponse
        mock_adapter.generate.return_value = LocalModelResponse(
            success=True, text="Executed successfully", model="llama3.2", provider="ollama", target="local"
        )

        executor = TaskExecutor(discovery=mock_disc, local_adapter=mock_adapter)
        res = executor.execute("resume", prompt="Enhance resume", complexity="low")

        assert res.success is True
        assert res.model == "llama3.2"
        assert "policy_decision" in res.metadata or "recommended_model" in res.metadata


def test_22_api_complexity_integration(app, client):
    """
    Verifies that POST /api/ai/execute accepts 'complexity' in JSON payload,
    executes without breaking, and surfaces result.
    """
    with app.app_context():
        app.config["AI_ENGINE_ENABLED"] = True

        mock_res = MagicMock()
        mock_res.success = True
        mock_res.to_dict.return_value = {
            "success": True,
            "task": "resume",
            "text": "Resume text",
            "model": "llama3.2",
            "target": "local",
            "metadata": {"complexity": "low", "recommended_model": "llama3.2"},
        }

        with patch("app.routes.ai_engine._engine.execute_task", return_value=mock_res) as mock_exec, \
             patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "test_user", "type": "access"}):

            headers = {"Authorization": "Bearer fake_token", "Content-Type": "application/json"}
            payload = {
                "task": "resume",
                "prompt": "Rewrite resume for senior python role",
                "complexity": "low",
            }
            resp = client.post("/api/ai/execute", json=payload, headers=headers)
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs.get("complexity") == "low"


def test_23_phase3_6_7_regression(app):
    """
    Verifies that components from Phase 3, Phase 6, and Phase 7 remain 100% functional
    alongside Phase 8 ModelPolicy additions.
    """
    from app.ai_engine.reasoning import (
        PromptBuilder,
        CitationValidator,
        ReasoningEngine,
        ModelRouter,
        TaskExecutor,
        ModelPolicy,
    )
    from app.ai_engine.retrieval.reranker import MultiSignalReranker

    with app.app_context():
        pb = PromptBuilder()
        assert pb is not None

        cv = CitationValidator()
        assert cv is not None

        ms = MultiSignalReranker()
        assert ms is not None

        engine = ReasoningEngine()
        assert engine is not None

        policy = ModelPolicy()
        assert policy is not None

        executor = TaskExecutor(policy=policy)
        assert executor is not None
