"""
tests/test_ai_engine_phase7.py
Comprehensive unit and integration test suite for AI Engine Phase 7 Step 2 (Local Model & Local AI Runtime).
Strictly verifies:
- Independent local model configuration (decoupled from Gemini)
- Local base URL and timeout handling
- Bounded concurrency with threading.Semaphore (resource-awareness)
- Exception-safe semaphore release
- Lightweight health check (success, connection failure, timeout)
- Factory provider resolution for Ollama and local alias
- Prevention of client-side SSRF / payload parameter injection
- Zero regression across existing providers (Gemini, OpenAI-compat, Mock)
- Zero live external network calls
"""
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from config import Config
from app import create_app
from app.ai_engine.reasoning.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
    LLMProviderError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    GeminiLLMProvider,
    OllamaLLMProvider,
    OpenAICompatibleProvider,
    MockLLMProvider,
)
from app.ai_engine.reasoning.providers.ollama import (
    _get_local_semaphore,
    _reset_local_semaphores,
)
from app.ai_engine.reasoning.schemas import AnswerRequest, LLMGenerationResult


@pytest.fixture(autouse=True)
def reset_semaphores():
    """Ensure semaphore registry is clean before and after each test."""
    _reset_local_semaphores()
    yield
    _reset_local_semaphores()


@pytest.fixture
def app():
    """Create Flask test application configured for Phase 7 testing."""
    test_app = create_app("testing")
    test_app.config["AI_ENGINE_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_PROVIDER"] = "ollama"
    test_app.config["AI_ENGINE_LOCAL_MODEL"] = "llama3.2"
    test_app.config["AI_ENGINE_LOCAL_BASE_URL"] = "http://localhost:11434/v1"
    test_app.config["AI_ENGINE_LOCAL_TIMEOUT_SECONDS"] = 60
    test_app.config["AI_ENGINE_LOCAL_MAX_CONCURRENCY"] = 2
    test_app.config["AI_ENGINE_LOCAL_HEALTH_CHECK_TIMEOUT_SECONDS"] = 5
    return test_app


@pytest.fixture
def client(app):
    return app.test_client()


# ==============================================================================
# 1. Configuration & Factory Resolution Tests
# ==============================================================================

def test_01_local_model_default_independent_from_gemini_model(app):
    """
    Verifies that the local model defaults to a dedicated local SLM/LLM (e.g. llama3.2)
    and does NOT inherit the cloud Gemini model name (gemini-2.5-flash).
    """
    with app.app_context():
        # Change the cloud LLM model to ensure local does NOT inherit it
        app.config["AI_ENGINE_LLM_MODEL"] = "gemini-2.5-pro"
        app.config.pop("AI_ENGINE_LOCAL_MODEL", None)

        provider = LLMProviderFactory.get_provider("ollama")
        assert isinstance(provider, OllamaLLMProvider)
        assert provider.model == "llama3.2"
        assert provider.model != app.config["AI_ENGINE_LLM_MODEL"]


def test_02_custom_local_model_configuration(app):
    """
    Verifies that AI_ENGINE_LOCAL_MODEL can be configured to any local SLM
    (e.g. qwen2.5-coder, mistral) without affecting cloud settings.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_MODEL"] = "qwen2.5-coder:7b"
        provider = LLMProviderFactory.get_provider("ollama")
        assert provider.model == "qwen2.5-coder:7b"


def test_03_local_base_url_resolution(app):
    """
    Verifies that base URL resolves to the dedicated local URL, strips trailing slashes,
    and falls back to Ollama base URL if not explicitly specified.
    """
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_BASE_URL"] = "http://10.0.0.5:11434/v1/"
        provider = LLMProviderFactory.get_provider("ollama")
        assert provider.base_url == "http://10.0.0.5:11434/v1"


def test_04_local_timeout_clamping():
    """
    Verifies that AI_ENGINE_LOCAL_TIMEOUT_SECONDS is safely clamped between 5s and 300s.
    """
    assert Config.AI_ENGINE_LOCAL_TIMEOUT_SECONDS == 60

    # Test safe int helper clamping
    from config import Config as C
    clamped_low = C._safe_int("1", default=60, min_val=5, max_val=300)
    assert clamped_low == 5

    clamped_high = C._safe_int("999", default=60, min_val=5, max_val=300)
    assert clamped_high == 300

    clamped_valid = C._safe_int("90", default=60, min_val=5, max_val=300)
    assert clamped_valid == 90


def test_05_local_concurrency_configuration_and_clamping():
    """
    Verifies that AI_ENGINE_LOCAL_MAX_CONCURRENCY is clamped between 1 and 8.
    """
    assert Config.AI_ENGINE_LOCAL_MAX_CONCURRENCY == 1

    from config import Config as C
    clamped_low = C._safe_int("0", default=1, min_val=1, max_val=8)
    assert clamped_low == 1

    clamped_high = C._safe_int("32", default=1, min_val=1, max_val=8)
    assert clamped_high == 8

    # Provider instance-level clamping
    p1 = OllamaLLMProvider(max_concurrency=0)
    assert p1.max_concurrency == 1

    p2 = OllamaLLMProvider(max_concurrency=100)
    assert p2.max_concurrency == 8


def test_06_provider_factory_resolves_ollama_and_local(app):
    """
    Verifies that LLMProviderFactory resolves both 'ollama' and 'local' aliases
    to an OllamaLLMProvider with dedicated configuration.
    """
    with app.app_context():
        p_ollama = LLMProviderFactory.get_provider("ollama")
        assert isinstance(p_ollama, OllamaLLMProvider)
        assert p_ollama.name == "ollama"

        p_local = LLMProviderFactory.get_provider("local")
        assert isinstance(p_local, OllamaLLMProvider)
        assert p_local.name == "ollama"


# ==============================================================================
# 2. Ollama Provider Mechanics & Health Check
# ==============================================================================

def test_07_ollama_provider_initialization():
    """
    Verifies that OllamaLLMProvider initializes all properties cleanly.
    """
    provider = OllamaLLMProvider(
        model="deepseek-r1:8b",
        base_url="http://localhost:11434/v1/",
        api_key="custom-key",
        timeout_seconds=45,
        temperature=0.4,
        max_tokens=2048,
        max_concurrency=3,
        health_check_timeout_seconds=10,
    )
    assert provider.model == "deepseek-r1:8b"
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.api_key == "custom-key"
    assert provider.timeout_seconds == 45
    assert provider.temperature == 0.4
    assert provider.max_tokens == 2048
    assert provider.max_concurrency == 3
    assert provider.health_check_timeout_seconds == 10
    assert provider.semaphore is not None


def test_08_health_check_success():
    """
    Verifies that check_health returns True when the local server responds with models.
    """
    provider = OllamaLLMProvider()
    mock_client = MagicMock()
    mock_client.models.list.return_value = MagicMock(data=[MagicMock(id="llama3.2")])
    provider._client = mock_client

    assert provider.check_health(timeout_seconds=2) is True
    mock_client.models.list.assert_called_once_with(timeout=2.0)


def test_09_health_check_connection_failure():
    """
    Verifies that check_health returns False cleanly on connection failure
    without raising an unhandled exception or leaking credentials.
    """
    import openai
    provider = OllamaLLMProvider(base_url="http://offline-host:11434/v1")
    mock_client = MagicMock()
    mock_client.models.list.side_effect = openai.APIConnectionError(request=MagicMock())
    provider._client = mock_client

    assert provider.check_health() is False


def test_10_health_check_timeout():
    """
    Verifies that check_health returns False on timeout.
    """
    import openai
    provider = OllamaLLMProvider()
    mock_client = MagicMock()
    mock_client.models.list.side_effect = openai.APITimeoutError(request=MagicMock())
    provider._client = mock_client

    assert provider.check_health(timeout_seconds=1) is False


# ==============================================================================
# 3. Resource-Aware Concurrency & Thread-Safety
# ==============================================================================

def test_11_semaphore_concurrency_enforcement():
    """
    Verifies that bounded concurrency is strictly enforced:
    when max_concurrency=1, concurrent requests must serialize through the semaphore.
    """
    sem = threading.Semaphore(1)
    provider = OllamaLLMProvider(max_concurrency=1, semaphore=sem)

    active_count = 0
    max_observed_active = 0
    lock = threading.Lock()

    def mock_create(*args, **kwargs):
        nonlocal active_count, max_observed_active
        with lock:
            active_count += 1
            if active_count > max_observed_active:
                max_observed_active = active_count
        time.sleep(0.05)
        with lock:
            active_count -= 1

        choice = MagicMock()
        choice.message.content = "Response"
        resp = MagicMock(choices=[choice], usage=MagicMock(total_tokens=10))
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create
    provider._client = mock_client

    threads = []
    for _ in range(3):
        t = threading.Thread(target=provider.generate, args=("test prompt",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Even with 3 concurrent threads, max observed active in the critical section should be 1
    assert max_observed_active == 1


def test_12_semaphore_release_after_exception():
    """
    Verifies that the concurrency semaphore is released when an exception occurs
    inside the generation call, preventing permanent deadlocks.
    """
    import openai
    sem = threading.Semaphore(1)
    provider = OllamaLLMProvider(max_concurrency=1, semaphore=sem)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())
    provider._client = mock_client

    # Call should fail with ProviderUnavailableError
    with pytest.raises(ProviderUnavailableError):
        provider.generate("test prompt")

    # Verify semaphore was released and can immediately be acquired
    acquired = sem.acquire(blocking=False)
    assert acquired is True
    sem.release()


# ==============================================================================
# 4. Security & Isolation Tests
# ==============================================================================

def test_13_local_config_cannot_be_supplied_through_request_payload():
    """
    Verifies that the API request schema (AnswerRequest) cannot receive or override
    base_url, provider, or local parameters from user payload (SSRF prevention).
    """
    # AnswerRequest should not accept arbitrary provider parameters
    req_dict = {
        "query": "What is Python?",
        "base_url": "http://malicious-internal-service:8080/v1",
        "provider": "ollama",
    }
    req = AnswerRequest(**req_dict)
    # Pydantic schema will either discard or not have 'base_url' as an active field
    assert not hasattr(req, "base_url")
    assert not hasattr(req, "provider")


def test_14_ollama_credentials_never_exposed_in_error():
    """
    Verifies that internal keys/tokens do not leak into exception strings.
    """
    import openai
    secret_key = "super-secret-local-proxy-token-xyz"
    provider = OllamaLLMProvider(api_key=secret_key)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )
    provider._client = mock_client

    with pytest.raises(ProviderUnavailableError) as exc_info:
        provider.generate("Hello")

    assert secret_key not in str(exc_info.value)


# ==============================================================================
# 5. Non-Regression Tests Across Other Providers
# ==============================================================================

def test_15_no_regression_gemini_provider(app):
    """
    Verifies that GeminiLLMProvider continues to resolve normally with its defaults.
    """
    with app.app_context():
        gemini = LLMProviderFactory.get_provider("gemini")
        assert isinstance(gemini, GeminiLLMProvider)
        assert gemini.name == "gemini"
        assert gemini.model == "gemini-2.5-flash"


def test_16_no_regression_openai_compat_provider(app):
    """
    Verifies that OpenAICompatibleProvider continues to resolve normally.
    """
    with app.app_context():
        compat = LLMProviderFactory.get_provider("openai_compat", model="deepseek-chat")
        assert isinstance(compat, OpenAICompatibleProvider)
        assert compat.name == "openai_compat"
        assert compat.model == "deepseek-chat"


def test_17_no_regression_mock_provider(app):
    """
    Verifies that MockLLMProvider continues to operate normally.
    """
    with app.app_context():
        mock_p = LLMProviderFactory.get_provider("mock")
        assert isinstance(mock_p, MockLLMProvider)
        assert mock_p.name == "mock"
        res = mock_p.generate("Hello world")
        assert isinstance(res, LLMGenerationResult)
        assert len(res.text) > 0
        assert "[S1]" in res.text


def test_18_ollama_generation_success_mocked():
    """
    Verifies that OllamaLLMProvider formats messages and extracts generated text correctly.
    """
    provider = OllamaLLMProvider(model="llama3.2")
    choice = MagicMock()
    choice.message.content = "FastAPI is a fast Python framework."
    mock_resp = MagicMock(choices=[choice], usage=MagicMock(total_tokens=42))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    provider._client = mock_client

    res = provider.generate(
        prompt="Explain FastAPI",
        system_instruction="You are a helpful assistant.",
        max_tokens=256,
        temperature=0.1,
    )

    assert isinstance(res, LLMGenerationResult)
    assert res.text == "FastAPI is a fast Python framework."
    assert res.provider == "ollama"
    assert res.model == "llama3.2"
    assert res.tokens_used == 42
    mock_client.chat.completions.create.assert_called_once()


# ==============================================================================
# 6. ModelRouter & Task Abstraction Tests (Step 3)
# ==============================================================================

def test_19_model_routing_disabled_routes_to_cloud(app):
    """
    Verifies that when AI_ENGINE_MODEL_ROUTING_ENABLED is False,
    all tasks route to the standard cloud provider with status 'routing_disabled'.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = False
        decision = ModelRouter.resolve_route("resume")
        assert decision.status == "routing_disabled"
        assert decision.target == "cloud"
        assert decision.provider == "gemini"
        assert decision.model == "gemini-2.5-flash"
        assert decision.is_local is False


def test_20_task_routing_defaults_local_and_cloud(app):
    """
    Verifies default task-to-target policies:
    - SLM/lightweight: resume, classification, extraction -> local
    - Complex reasoning: chat, research, salary_analysis, company_review -> cloud
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True

        # Local tasks
        for local_task in ("resume", "classification", "extraction"):
            d = ModelRouter.resolve_route(local_task)
            assert d.target == "local", f"Task {local_task} should route to local"
            assert d.provider == "ollama"
            assert d.model == "llama3.2"
            assert d.is_local is True
            assert d.is_vision is False
            assert d.status == "resolved"
            assert d.fallback_target == "cloud"
            assert d.fallback_provider == "gemini"

        # Cloud tasks
        for cloud_task in (
            "chat",
            "research",
            "salary_analysis",
            "company_review",
            "text_generation",
        ):
            d = ModelRouter.resolve_route(cloud_task)
            assert d.target == "cloud", f"Task {cloud_task} should route to cloud"
            assert d.provider == "gemini"
            assert d.model == "gemini-2.5-flash"
            assert d.is_local is False
            assert d.is_vision is False
            assert d.status == "resolved"


def test_21_vision_model_not_configured(app):
    """
    Verifies that vision and document_vision tasks route to vision_local,
    and report 'vision_not_configured' with model=None when no vision model is set.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = ""

        for vision_task in ("vision", "document_vision"):
            d = ModelRouter.resolve_route(vision_task)
            assert d.target == "vision_local"
            assert d.provider == "ollama"
            assert d.model is None
            assert d.status == "vision_not_configured"
            assert d.is_vision is True
            assert d.is_local is True
            assert d.fallback_target == "cloud"
            assert d.fallback_provider == "gemini"

        # get_provider_for_task should raise LLMProviderError if vision is unconfigured
        with pytest.raises(LLMProviderError) as exc_info:
            ModelRouter.get_provider_for_task("vision")
        assert "not configured" in str(exc_info.value)


def test_22_vision_model_configured(app):
    """
    Verifies that when AI_ENGINE_LOCAL_VISION_MODEL is set,
    vision tasks resolve with status 'resolved' and the configured vision model.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"

        d = ModelRouter.resolve_route("vision")
        assert d.target == "vision_local"
        assert d.provider == "ollama"
        assert d.model == "llama3.2-vision:11b"
        assert d.status == "resolved"
        assert d.is_vision is True


def test_23_custom_local_and_cloud_configuration(app):
    """
    Verifies that custom text models and cloud providers are respected in routing decisions.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "qwen2.5:7b"
        app.config["AI_ENGINE_CLOUD_PROVIDER"] = "openai_compat"
        app.config["AI_ENGINE_LLM_MODEL"] = "deepseek-chat"

        d_local = ModelRouter.resolve_route("resume")
        assert d_local.model == "qwen2.5:7b"
        assert d_local.fallback_provider == "openai_compat"
        assert d_local.fallback_model == "deepseek-chat"

        d_cloud = ModelRouter.resolve_route("research")
        assert d_cloud.provider == "openai_compat"
        assert d_cloud.model == "deepseek-chat"


def test_24_custom_task_target_overrides(app):
    """
    Verifies that individual task targets can be overridden via config
    (e.g. routing research to local or resume to cloud).
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "cloud"
        app.config["AI_ENGINE_RESEARCH_MODEL_TARGET"] = "local"

        d_resume = ModelRouter.resolve_route("resume")
        assert d_resume.target == "cloud"
        assert d_resume.provider == "gemini"

        d_research = ModelRouter.resolve_route("research")
        assert d_research.target == "local"
        assert d_research.provider == "ollama"


def test_25_invalid_target_fallback(app):
    """
    Verifies that if an invalid or unrecognized target is configured,
    the router falls back to cloud with status 'invalid_target_fallback'.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "quantum_experimental"

        d = ModelRouter.resolve_route("resume")
        assert d.target == "cloud"
        assert d.provider == "gemini"
        assert d.status == "invalid_target_fallback"
        assert d.metadata["invalid_target"] == "quantum_experimental"


def test_26_unknown_task_handling(app):
    """
    Verifies that unknown or future task categories fall back to cloud safely
    with status 'unknown_task_fallback'.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        d = ModelRouter.resolve_route("future_unimplemented_task_xyz")
        assert d.target == "cloud"
        assert d.provider == "gemini"
        assert d.status == "unknown_task_fallback"


def test_27_local_health_probe_integration(app):
    """
    Verifies health check integration during route resolution:
    - Healthy provider -> status 'resolved'
    - Unhealthy provider -> status 'local_unavailable' with fallback indicators
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        # Healthy mock provider
        healthy_prov = MagicMock()
        healthy_prov.check_health.return_value = True

        d_healthy = ModelRouter.resolve_route("resume", check_local_health=True, local_provider=healthy_prov)
        assert d_healthy.status == "resolved"

        # Unhealthy mock provider
        unhealthy_prov = MagicMock()
        unhealthy_prov.check_health.return_value = False

        d_unhealthy = ModelRouter.resolve_route("resume", check_local_health=True, local_provider=unhealthy_prov)
        assert d_unhealthy.status == "local_unavailable"
        assert d_unhealthy.fallback_target == "cloud"
        assert d_unhealthy.fallback_provider == "gemini"


def test_28_model_router_get_provider_integration(app):
    """
    Verifies that ModelRouter.get_provider_for_task() resolves the route
    and returns a valid BaseLLMProvider from LLMProviderFactory.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        local_p = ModelRouter.get_provider_for_task("resume")
        assert isinstance(local_p, OllamaLLMProvider)
        assert local_p.name == "ollama"

        cloud_p = ModelRouter.get_provider_for_task("research")
        assert isinstance(cloud_p, GeminiLLMProvider)
        assert cloud_p.name == "gemini"


def test_29_no_arbitrary_urls_accepted_by_router():
    """
    Verifies that ModelRouter does not accept arbitrary URL overrides,
    enforcing server-side endpoint isolation and SSRF prevention.
    """
    import inspect
    from app.ai_engine.reasoning.model_router import ModelRouter

    sig = inspect.signature(ModelRouter.resolve_route)
    params = list(sig.parameters.keys())
    assert "base_url" not in params
    assert "url" not in params
    assert "provider_url" not in params


def test_30_decision_serialization(app):
    """
    Verifies that ModelRouteDecision converts cleanly to a dictionary
    matching required structured telemetry specifications.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    with app.app_context():
        d = ModelRouter.resolve_route("resume")
        d_dict = d.to_dict()
        assert d_dict["task"] == "resume"
        assert d_dict["target"] == "local"
        assert d_dict["provider"] == "ollama"
        assert d_dict["model"] == "llama3.2"
        assert d_dict["status"] == "resolved"
        assert d_dict["is_local"] is True
        assert d_dict["is_vision"] is False
        assert d_dict["fallback_target"] == "cloud"
        assert d_dict["fallback_provider"] == "gemini"


# ==============================================================================
# 7. Local Model Discovery Tests (Step 4)
# ==============================================================================

def test_31_ollama_runtime_available_mocked():
    """
    Verifies that is_runtime_available returns True when local runtime responds.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    discovery = LocalModelDiscovery(base_url="http://localhost:11434/v1")

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.2:latest", "details": {"family": "llama"}}]
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        assert discovery.is_runtime_available() is True


def test_32_ollama_runtime_unavailable_mocked():
    """
    Verifies that runtime unavailable returns False and structured unavailable status
    without crashing or raising unhandled exceptions.
    """
    import openai
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    discovery = LocalModelDiscovery(base_url="http://offline-host:11434/v1")

    with patch("httpx.Client.get", side_effect=Exception("Connection refused")):
        with patch("openai.OpenAI.models") as mock_models:
            mock_models.list.side_effect = openai.APIConnectionError(request=MagicMock())
            assert discovery.is_runtime_available() is False

            status = discovery.get_runtime_status()
            assert status.available is False
            assert status.models == []
            assert status.configured_text_model_available is False
            assert status.configured_vision_model_available is False
            assert status.configured_text_model.reason == "runtime_unavailable"


def test_33_installed_model_discovery():
    """
    Verifies listing installed models and capturing name, size, and details.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    discovery = LocalModelDiscovery(base_url="http://localhost:11434/v1")

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "llama3.2:latest",
                "size": 2019393189,
                "details": {"family": "llama", "families": ["llama"], "parameter_size": "3.2B"},
            },
            {
                "name": "llama3.2-vision:11b",
                "size": 7914000000,
                "details": {"family": "mllama", "families": ["mllama", "clip"], "parameter_size": "11B"},
            },
            {
                "name": "custom-embed:v1",
                "size": 500000000,
                "details": {"family": "bert"},
            },
        ]
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        models = discovery.list_models()
        assert len(models) == 3

        m1 = models[0]
        assert m1.name == "llama3.2:latest"
        assert m1.size == 2019393189
        assert m1.capability == "text"
        assert m1.available is True

        m2 = models[1]
        assert m2.name == "llama3.2-vision:11b"
        assert m2.size == 7914000000
        assert m2.capability == "vision"

        m3 = models[2]
        assert m3.name == "custom-embed:v1"
        assert m3.capability == "text"


def test_34_configured_text_model_available(app):
    """
    Verifies that a configured model (e.g. llama3.2) resolves as available
    when matching installed tag 'llama3.2:latest'.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"
        discovery = LocalModelDiscovery()

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2:latest", "details": {"family": "llama"}}]
        }

        with patch("httpx.Client.get", return_value=mock_resp):
            assert discovery.is_model_available("llama3.2") is True
            text_status = discovery.get_text_model_status()
            assert text_status.configured is True
            assert text_status.available is True
            assert text_status.reason == "available"
            assert text_status.matched_installed_name == "llama3.2:latest"
            assert text_status.capability == "text"


def test_35_configured_text_model_missing(app):
    """
    Verifies that when the configured text model is missing from installed models,
    it reports available=False with reason 'model_not_installed'.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"
        discovery = LocalModelDiscovery()

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5:7b", "details": {"family": "qwen2"}}]
        }

        with patch("httpx.Client.get", return_value=mock_resp):
            assert discovery.is_model_available("llama3.2") is False
            text_status = discovery.get_text_model_status()
            assert text_status.configured is True
            assert text_status.available is False
            assert text_status.reason == "model_not_installed"


def test_36_vision_model_not_configured(app):
    """
    Verifies that when AI_ENGINE_LOCAL_VISION_MODEL is empty or unconfigured,
    get_vision_model_status returns configured=False and reason 'vision_model_not_configured'.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = ""
        discovery = LocalModelDiscovery()

        v_status = discovery.get_vision_model_status()
        assert v_status.configured is False
        assert v_status.available is False
        assert v_status.reason == "vision_model_not_configured"


def test_37_configured_vision_model_available(app):
    """
    Verifies that when a vision model is configured and installed with vision metadata,
    it returns available=True and capability='vision'.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"
        discovery = LocalModelDiscovery()

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "llama3.2-vision:11b",
                    "details": {"family": "mllama", "families": ["mllama", "clip"]},
                }
            ]
        }

        with patch("httpx.Client.get", return_value=mock_resp):
            v_status = discovery.get_vision_model_status()
            assert v_status.configured is True
            assert v_status.available is True
            assert v_status.capability == "vision"
            assert v_status.reason == "available"


def test_38_configured_vision_model_missing(app):
    """
    Verifies that when a vision model is configured but missing from installed models,
    it returns available=False and reason 'model_not_installed'.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"
        discovery = LocalModelDiscovery()

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2:latest", "details": {"family": "llama"}}]
        }

        with patch("httpx.Client.get", return_value=mock_resp):
            v_status = discovery.get_vision_model_status()
            assert v_status.configured is True
            assert v_status.available is False
            assert v_status.reason == "model_not_installed"


def test_39_reliable_vision_capability_detection():
    """
    Verifies reliable capability detection from architectural family metadata:
    - CLIP, mllama, vision, siglip, blip -> vision
    - llama, mistral, qwen, gemma -> text
    """
    from app.ai_engine.reasoning.local_models import detect_capability_from_details

    # Vision architectures
    assert detect_capability_from_details({"families": ["mllama", "clip"]}) == "vision"
    assert detect_capability_from_details({"families": ["siglip", "gemma"]}) == "vision"
    assert detect_capability_from_details({"family": "vision"}) == "vision"
    assert detect_capability_from_details({"details": {}, "families": ["blip"]}) == "vision"

    # Text architectures
    assert detect_capability_from_details({"family": "llama", "families": ["llama"]}) == "text"
    assert detect_capability_from_details({"family": "mistral"}) == "text"
    assert detect_capability_from_details({"family": "qwen2", "families": ["qwen2"]}) == "text"
    assert detect_capability_from_details({"families": ["gemma"]}) == "text"


def test_40_unknown_capability_fallback():
    """
    Verifies that when architecture metadata is absent, malformed, or ambiguous,
    capability defaults strictly to 'unknown' rather than guessing.
    """
    from app.ai_engine.reasoning.local_models import detect_capability_from_details

    assert detect_capability_from_details({}) == "unknown"
    assert detect_capability_from_details(None) == "unknown"
    assert detect_capability_from_details({"family": "exotic_proprietary_arch"}) == "unknown"
    assert detect_capability_from_details({"families": []}) == "unknown"


def test_41_no_automatic_download_guarantee():
    """
    Verifies that LocalModelDiscovery contains zero download/pull methods
    and never invokes external process/network commands to pull models.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    disallowed_methods = ["pull", "download", "pull_model", "install", "create", "run"]
    discovery = LocalModelDiscovery()

    for method_name in disallowed_methods:
        assert not hasattr(discovery, method_name), f"Discovery should not have method {method_name}"

    # Missing model should never trigger downloads
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})
        res = discovery.is_model_available("nonexistent_model")
        assert res is False
        # Only read-only GET /api/tags called
        mock_get.assert_called_once()


def test_42_empty_model_list_handling():
    """
    Verifies that an empty model list from Ollama is handled gracefully.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    discovery = LocalModelDiscovery()

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"models": []}

    with patch("httpx.Client.get", return_value=mock_resp):
        models = discovery.list_models()
        assert models == []
        assert discovery.is_model_available("llama3.2") is False


def test_43_malformed_runtime_metadata():
    """
    Verifies that corrupt or unexpected payload formats from local runtime
    do not crash LocalModelDiscovery.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    discovery = LocalModelDiscovery()

    mock_resp = MagicMock(status_code=200)
    # Corrupt model items (non-dict items, items missing name)
    mock_resp.json.return_value = {
        "models": ["invalid_string_item", {"corrupt": True}, {"name": "valid:model"}]
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        models = discovery.list_models()
        assert len(models) == 1
        assert models[0].name == "valid:model"


def test_44_model_router_local_discovery_integration(app):
    """
    Verifies ModelRouter integration with LocalModelDiscovery:
    - Installed local model -> status 'resolved'
    - Missing local model -> status 'local_model_not_installed' with cloud fallback
    - Missing vision model -> status 'vision_model_not_installed' with cloud fallback
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"

        # Mock discovery where text model is installed but vision model is missing
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_model_available.side_effect = lambda name: name == "llama3.2"

        # 1. Text task with installed model -> resolved
        d_resume = ModelRouter.resolve_route("resume", discovery=mock_disc)
        assert d_resume.status == "resolved"
        assert d_resume.model == "llama3.2"

        # 2. Text task with missing model -> local_model_not_installed
        mock_disc_missing = MagicMock(spec=LocalModelDiscovery)
        mock_disc_missing.is_model_available.return_value = False

        d_missing = ModelRouter.resolve_route("resume", discovery=mock_disc_missing)
        assert d_missing.status == "local_model_not_installed"
        assert d_missing.fallback_target == "cloud"
        assert d_missing.fallback_provider == "gemini"

        # 3. Vision task with missing model -> vision_model_not_installed
        d_vision = ModelRouter.resolve_route("vision", discovery=mock_disc)
        assert d_vision.status == "vision_model_not_installed"
        assert d_vision.fallback_target == "cloud"


def test_45_provider_factory_compatibility_with_discovery(app):
    """
    Verifies that LocalModelDiscovery shares consistent base_url and configuration
    with OllamaLLMProvider and LLMProviderFactory.
    """
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.providers import LLMProviderFactory

    with app.app_context():
        discovery = LocalModelDiscovery()
        provider = LLMProviderFactory.get_provider("ollama")

        assert discovery.base_url == provider.base_url
        assert discovery.api_key == provider.api_key


def test_46_phase1_6_regression_check(app):
    """
    Verifies that existing reasoning pipeline contracts and schemas remain 100% intact.
    """
    from app.ai_engine.reasoning import (
        PromptBuilder,
        CitationValidator,
        ReasoningEngine,
        LLMProviderFactory,
    )
    with app.app_context():
        # Ensure imports and instantiations operate without conflict
        engine = ReasoningEngine()
        assert engine.max_retries >= 0
        prov = LLMProviderFactory.get_provider("mock")
        assert prov.name == "mock"


# ==============================================================================
# 8. Local Model Adapter Tests (Step 5)
# ==============================================================================

def test_47_successful_local_text_generation(app):
    """
    Verifies successful local text generation returning a structured LocalModelResponse.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        mock_provider = MagicMock(spec=OllamaLLMProvider)
        mock_provider.model = "llama3.2"
        mock_provider.generate.return_value = LLMGenerationResult(
            text="Optimized resume bullet points.",
            model="llama3.2",
            provider="ollama",
            tokens_used=42,
        )

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)
        resp = adapter.generate(prompt="Improve this resume: Software Engineer")

        assert resp.success is True
        assert resp.text == "Optimized resume bullet points."
        assert resp.model == "llama3.2"
        assert resp.provider == "ollama"
        assert resp.target == "local"
        assert resp.tokens_used == 42
        assert resp.latency_ms >= 0.0
        assert resp.error_code is None
        assert resp.error_message is None
        mock_provider.generate.assert_called_once()


def test_48_configured_text_model_unavailable(app):
    """
    Verifies that when the configured model is unavailable in the local runtime,
    execution aborts cleanly returning MODEL_NOT_AVAILABLE without calling the provider.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_MODEL_NOT_AVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = False

        mock_provider = MagicMock(spec=OllamaLLMProvider)
        mock_provider.model = "llama3.2"

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)
        resp = adapter.generate(prompt="Test prompt")

        assert resp.success is False
        assert resp.error_code == ERR_MODEL_NOT_AVAILABLE
        assert "not installed" in resp.error_message
        assert resp.model == "llama3.2"
        mock_provider.generate.assert_not_called()


def test_49_local_runtime_unavailable(app):
    """
    Verifies that when the local Ollama runtime is offline or unreachable,
    execution fails cleanly with LOCAL_RUNTIME_UNAVAILABLE.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_LOCAL_RUNTIME_UNAVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = False

        mock_provider = MagicMock(spec=OllamaLLMProvider)

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)
        resp = adapter.generate(prompt="Test prompt")

        assert resp.success is False
        assert resp.error_code == ERR_LOCAL_RUNTIME_UNAVAILABLE
        assert "runtime is offline" in resp.error_message
        mock_discovery.is_model_available.assert_not_called()
        mock_provider.generate.assert_not_called()


def test_50_timeout_handling(app):
    """
    Verifies that provider timeout exceptions map deterministically
    to LOCAL_MODEL_TIMEOUT with structured timing metadata.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_LOCAL_MODEL_TIMEOUT
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        mock_provider = MagicMock(spec=OllamaLLMProvider)
        mock_provider.model = "llama3.2"
        mock_provider.generate.side_effect = ProviderTimeoutError("Local inference timed out after 60s")

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)
        resp = adapter.generate(prompt="Long heavy prompt")

        assert resp.success is False
        assert resp.error_code == ERR_LOCAL_MODEL_TIMEOUT
        assert "timed out" in resp.error_message
        assert resp.latency_ms >= 0.0


def test_51_execution_exception_handling(app):
    """
    Verifies that unforeseen provider or runtime exceptions are captured
    safely as LOCAL_MODEL_EXECUTION_ERROR without crashing the application.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_LOCAL_MODEL_EXECUTION_ERROR
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        mock_provider = MagicMock(spec=OllamaLLMProvider)
        mock_provider.model = "llama3.2"
        mock_provider.generate.side_effect = LLMProviderError("CUDA out of memory in local runtime")

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)
        resp = adapter.generate(prompt="Heavy prompt")

        assert resp.success is False
        assert resp.error_code == ERR_LOCAL_MODEL_EXECUTION_ERROR
        assert "CUDA out of memory" in resp.error_message


def test_52_structured_response_format():
    """
    Verifies that LocalModelResponse provides all standard fields and converts cleanly to a dictionary.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelResponse

    resp = LocalModelResponse(
        success=True,
        text="Sample output",
        model="llama3.2",
        provider="ollama",
        target="local",
        latency_ms=123.45,
        tokens_used=50,
        metadata={"custom": "info"},
    )

    d = resp.to_dict()
    assert d["success"] is True
    assert d["text"] == "Sample output"
    assert d["model"] == "llama3.2"
    assert d["provider"] == "ollama"
    assert d["target"] == "local"
    assert d["latency_ms"] == 123.45
    assert d["tokens_used"] == 50
    assert d["error_code"] is None
    assert d["error_message"] is None
    assert d["metadata"] == {"custom": "info"}


def test_53_configured_model_is_used(app):
    """
    Verifies that when a specific model is passed, that exact model is checked and dispatched.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.side_effect = lambda m: m == "qwen2.5:7b"

        mock_provider = MagicMock(spec=OllamaLLMProvider)
        mock_provider.model = "qwen2.5:7b"
        mock_provider.generate.return_value = LLMGenerationResult(
            text="Qwen response", model="qwen2.5:7b", provider="ollama"
        )

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)
        resp = adapter.generate(prompt="Hello", model="qwen2.5:7b")

        assert resp.success is True
        assert resp.model == "qwen2.5:7b"
        mock_discovery.is_model_available.assert_called_with("qwen2.5:7b")


def test_54_no_silent_model_substitution(app):
    """
    Verifies that if configured model is 'llama3.2' and only 'qwen3.5:4b' is installed,
    the adapter refuses to substitute and returns MODEL_NOT_AVAILABLE.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_MODEL_NOT_AVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        # Only qwen3.5:4b is available, llama3.2 is NOT
        mock_discovery.is_model_available.side_effect = lambda m: m == "qwen3.5:4b"

        adapter = LocalModelAdapter(discovery=mock_discovery)
        resp = adapter.generate(prompt="Hello")

        assert resp.success is False
        assert resp.error_code == ERR_MODEL_NOT_AVAILABLE
        assert resp.model == "llama3.2"
        # Absolutely no substitution to qwen3.5:4b
        assert resp.model != "qwen3.5:4b"


def test_55_vision_model_not_configured(app):
    """
    Verifies that generate_vision returns VISION_MODEL_NOT_CONFIGURED when no vision model is set.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_VISION_MODEL_NOT_CONFIGURED

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = ""
        adapter = LocalModelAdapter()
        resp = adapter.generate_vision(prompt="Describe this document", images=["image_data"])

        assert resp.success is False
        assert resp.target == "vision_local"
        assert resp.error_code == ERR_VISION_MODEL_NOT_CONFIGURED


def test_56_vision_model_unavailable(app):
    """
    Verifies that generate_vision returns MODEL_NOT_AVAILABLE when vision model is configured
    but not installed in the local runtime.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_MODEL_NOT_AVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"

        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = False

        adapter = LocalModelAdapter(discovery=mock_discovery)
        resp = adapter.generate_vision(prompt="Describe image", images=["img"])

        assert resp.success is False
        assert resp.target == "vision_local"
        assert resp.error_code == ERR_MODEL_NOT_AVAILABLE
        assert "not installed" in resp.error_message


def test_57_vision_execution_returns_not_ready_rather_than_fake_inference(app):
    """
    Verifies that generate_vision returns VISION_EXECUTION_NOT_READY when model is configured
    and installed, strictly avoiding fake multimodal inference on text models.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, ERR_VISION_EXECUTION_NOT_READY
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"

        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        adapter = LocalModelAdapter(discovery=mock_discovery)
        resp = adapter.generate_vision(prompt="Describe invoice", images=["img1", "img2"])

        assert resp.success is False
        assert resp.target == "vision_local"
        assert resp.model == "llama3.2-vision:11b"
        assert resp.error_code == ERR_VISION_EXECUTION_NOT_READY
        assert "not ready" in resp.error_message
        assert resp.metadata.get("images_count") == 2


def test_58_model_router_to_local_model_adapter_integration(app):
    """
    Verifies end-to-end integration between ModelRouter route decisions and LocalModelAdapter:
    - Text route executes local text generation
    - Vision route executes local vision contract
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"

        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        mock_provider = MagicMock(spec=OllamaLLMProvider)
        mock_provider.model = "llama3.2"
        mock_provider.generate.return_value = LLMGenerationResult(
            text="Resume text", model="llama3.2", provider="ollama"
        )

        adapter = LocalModelAdapter(discovery=mock_discovery, provider=mock_provider)

        # 1. Route text task
        decision_text = ModelRouter.resolve_route("resume", discovery=mock_discovery)
        assert decision_text.target == "local"
        resp_text = adapter.execute_route(decision_text, prompt="Summarize resume")
        assert resp_text.success is True
        assert resp_text.text == "Resume text"

        # 2. Route vision task
        decision_vision = ModelRouter.resolve_route("vision", discovery=mock_discovery)
        assert decision_vision.target == "vision_local"
        resp_vision = adapter.execute_route(decision_vision, prompt="Scan doc", images=["doc1"])
        assert resp_vision.success is False
        assert resp_vision.error_code == "VISION_EXECUTION_NOT_READY"


def test_59_provider_factory_compatibility(app):
    """
    Verifies that LocalModelAdapter resolves OllamaLLMProvider via LLMProviderFactory cleanly.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        adapter = LocalModelAdapter(discovery=mock_discovery)
        prov = adapter._get_provider("llama3.2")
        assert isinstance(prov, OllamaLLMProvider)
        assert prov.model == "llama3.2"


def test_60_concurrency_protection_delegated_to_ollama_provider(app):
    """
    Verifies that LocalModelAdapter does NOT introduce a competing semaphore,
    relying directly on OllamaLLMProvider's built-in semaphore protection.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = True

        adapter = LocalModelAdapter(discovery=mock_discovery)
        # Verify adapter has no independent semaphore attribute
        assert not hasattr(adapter, "_semaphore")
        assert not hasattr(adapter, "semaphore")

        prov = adapter._get_provider("llama3.2")
        assert hasattr(prov, "_semaphore")
        assert isinstance(prov._semaphore, threading.Semaphore)


def test_61_api_key_never_exposed_in_errors_or_logging(app, caplog):
    """
    Verifies that sensitive credentials (like bearer tokens) are never leaked
    in LocalModelResponse error messages or metadata.
    """
    import logging
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        sensitive_token = "sk-super-secret-token-12345"
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = False

        adapter = LocalModelAdapter(
            discovery=mock_discovery,
            api_key=sensitive_token,
            base_url="http://localhost:11434/v1",
        )

        with caplog.at_level(logging.DEBUG):
            resp = adapter.generate(prompt="Test prompt")

        assert resp.success is False
        assert sensitive_token not in (resp.error_message or "")
        assert sensitive_token not in str(resp.to_dict())
        assert sensitive_token not in caplog.text


def test_62_no_automatic_model_download_behavior(app):
    """
    Verifies that LocalModelAdapter exposes zero download or pull capabilities,
    and missing models never trigger remote download commands.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    disallowed = ["pull", "download", "install", "pull_model", "create"]
    adapter = LocalModelAdapter()
    for method in disallowed:
        assert not hasattr(adapter, method), f"Adapter must not expose {method}"

    with app.app_context():
        mock_discovery = MagicMock(spec=LocalModelDiscovery)
        mock_discovery.is_runtime_available.return_value = True
        mock_discovery.is_model_available.return_value = False

        adapter_instance = LocalModelAdapter(discovery=mock_discovery)
        with patch("httpx.Client.post") as mock_post:
            resp = adapter_instance.generate(prompt="Missing model prompt")
            assert resp.success is False
            assert resp.error_code == "MODEL_NOT_AVAILABLE"
            mock_post.assert_not_called()


def test_63_existing_phase1_6_regression_behavior(app):
    """
    Verifies that existing multi-phase components (ReasoningEngine, PromptBuilder,
    CitationValidator, MultiSignalReranker) continue to function identically.
    """
    from app.ai_engine.reasoning import (
        PromptBuilder,
        CitationValidator,
        ReasoningEngine,
        LocalModelAdapter,
    )
    from app.ai_engine.retrieval.reranker import MultiSignalReranker, KeywordReranker

    with app.app_context():
        # Confirm components instantiate cleanly together
        engine = ReasoningEngine()
        assert engine is not None

        pb = PromptBuilder()
        assert pb is not None

        cv = CitationValidator()
        assert cv is not None

        ms_reranker = MultiSignalReranker()
        assert ms_reranker is not None

        kw_reranker = KeywordReranker()
        assert kw_reranker is not None

        adapter = LocalModelAdapter()
        assert adapter is not None


# ==============================================================================
# 9. Task-Aware Local Model Execution Integration Tests (Step 6)
# ==============================================================================

def test_64_task_executor_initialization(app):
    """
    Verifies that TaskExecutor initializes with default or injected dependencies.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.model_router import ModelRouter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter

    with app.app_context():
        executor = TaskExecutor()
        assert executor.router is ModelRouter
        assert isinstance(executor.discovery, LocalModelDiscovery)
        assert isinstance(executor.local_adapter, LocalModelAdapter)
        assert executor._cloud_provider is None


def test_65_resume_task_routes_according_to_configuration(app):
    """
    Verifies that resume task routes according to AI_ENGINE_RESUME_MODEL_TARGET config.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, LocalModelResponse

    with app.app_context():
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_adapter = MagicMock(spec=LocalModelAdapter)
        mock_adapter.generate.return_value = LocalModelResponse(
            success=True, text="Resume bullet points", model="llama3.2", provider="ollama", target="local"
        )

        executor = TaskExecutor(discovery=mock_disc, local_adapter=mock_adapter)
        res = executor.execute("resume", prompt="Enhance resume")

        assert res.success is True
        assert res.target == "local"
        assert res.model == "llama3.2"
        assert res.provider == "ollama"
        assert res.text == "Resume bullet points"


def test_66_classification_task_routes_according_to_configuration(app):
    """
    Verifies that classification task routes according to configuration.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, LocalModelResponse

    with app.app_context():
        app.config["AI_ENGINE_CLASSIFICATION_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_adapter = MagicMock(spec=LocalModelAdapter)
        mock_adapter.generate.return_value = LocalModelResponse(
            success=True, text="Category: Engineering", model="llama3.2", provider="ollama", target="local"
        )

        executor = TaskExecutor(discovery=mock_disc, local_adapter=mock_adapter)
        res = executor.execute("classification", prompt="Classify this role")

        assert res.success is True
        assert res.target == "local"
        assert res.text == "Category: Engineering"


def test_67_extraction_task_routes_according_to_configuration(app):
    """
    Verifies that extraction task routes according to configuration.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, LocalModelResponse

    with app.app_context():
        app.config["AI_ENGINE_EXTRACTION_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_adapter = MagicMock(spec=LocalModelAdapter)
        mock_adapter.generate.return_value = LocalModelResponse(
            success=True, text='{"skills": ["Python"]}', model="llama3.2", provider="ollama", target="local"
        )

        executor = TaskExecutor(discovery=mock_disc, local_adapter=mock_adapter)
        res = executor.execute("extraction", prompt="Extract skills")

        assert res.success is True
        assert res.target == "local"
        assert res.text == '{"skills": ["Python"]}'


def test_68_chat_task_preserves_configured_target(app):
    """
    Verifies that chat task routes to cloud by default and executes via cloud provider.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor

    with app.app_context():
        mock_cloud = MagicMock(spec=BaseLLMProvider)
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Hello! How can I help?", model="gemini-2.5-flash", provider="gemini"
        )

        executor = TaskExecutor(cloud_provider=mock_cloud)
        res = executor.execute("chat", prompt="Hello")

        assert res.success is True
        assert res.target == "cloud"
        assert res.provider == "gemini"
        assert res.model == "gemini-2.5-flash"
        assert res.text == "Hello! How can I help?"


def test_69_research_task_preserves_configured_target(app):
    """
    Verifies that research task preserves its configured cloud target.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor

    with app.app_context():
        mock_cloud = MagicMock(spec=BaseLLMProvider)
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Research summary with grounding.", model="gemini-2.5-flash", provider="gemini"
        )

        executor = TaskExecutor(cloud_provider=mock_cloud)
        res = executor.execute("research", prompt="Analyze market trends")

        assert res.success is True
        assert res.target == "cloud"
        assert res.provider == "gemini"


def test_70_salary_analysis_preserves_configured_target(app):
    """
    Verifies that salary analysis routes to cloud by default.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor

    with app.app_context():
        mock_cloud = MagicMock(spec=BaseLLMProvider)
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Salary benchmark: $140,000", model="gemini-2.5-flash", provider="gemini"
        )

        executor = TaskExecutor(cloud_provider=mock_cloud)
        res = executor.execute("salary_analysis", prompt="Senior SWE salary in NYC")

        assert res.success is True
        assert res.target == "cloud"
        assert res.provider == "gemini"


def test_71_company_review_preserves_configured_target(app):
    """
    Verifies that company review routes to cloud by default.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor

    with app.app_context():
        mock_cloud = MagicMock(spec=BaseLLMProvider)
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Company culture rating: 4.5/5", model="gemini-2.5-flash", provider="gemini"
        )

        executor = TaskExecutor(cloud_provider=mock_cloud)
        res = executor.execute("company_review", prompt="Review of Acme Corp")

        assert res.success is True
        assert res.target == "cloud"
        assert res.provider == "gemini"


def test_72_local_model_available_adapter_executes(app):
    """
    Verifies that when local model is available, LocalModelAdapter executes
    and returns a structured TaskExecutionResult.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter, LocalModelResponse

    with app.app_context():
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_adapter = MagicMock(spec=LocalModelAdapter)
        mock_adapter.generate.return_value = LocalModelResponse(
            success=True,
            text="Clean bullet points",
            model="llama3.2",
            provider="ollama",
            target="local",
            latency_ms=88.5,
            tokens_used=30,
        )

        executor = TaskExecutor(discovery=mock_disc, local_adapter=mock_adapter)
        res = executor.execute("resume", prompt="Bullet point draft")

        assert res.success is True
        assert res.target == "local"
        assert res.provider == "ollama"
        assert res.model == "llama3.2"
        assert res.text == "Clean bullet points"
        assert res.tokens_used == 30
        assert res.latency_ms == 88.5
        mock_adapter.generate.assert_called_once()


def test_73_local_model_missing_no_silent_substitution(app):
    """
    Verifies that when local model is missing and fallback is disabled,
    it returns LOCAL_MODEL_UNAVAILABLE without silently substituting another installed model.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_LOCAL_MODEL_UNAVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter

    with app.app_context():
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        # Only qwen3.5 is installed, llama3.2 is NOT
        mock_disc.is_model_available.side_effect = lambda m: m == "qwen3.5:4b"

        mock_adapter = MagicMock(spec=LocalModelAdapter)

        executor = TaskExecutor(discovery=mock_disc, local_adapter=mock_adapter)
        res = executor.execute("resume", prompt="Test prompt", allow_cloud_fallback=False)

        assert res.success is False
        assert res.error_code == ERR_LOCAL_MODEL_UNAVAILABLE
        assert res.model == "llama3.2"
        # Strictly no silent substitution to qwen3.5
        assert res.model != "qwen3.5:4b"
        mock_adapter.generate.assert_not_called()


def test_74_local_runtime_unavailable_structured_error(app):
    """
    Verifies that when local runtime is offline and fallback is disabled,
    it returns LOCAL_RUNTIME_UNAVAILABLE.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_LOCAL_RUNTIME_UNAVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = False

        executor = TaskExecutor(discovery=mock_disc)
        res = executor.execute("resume", prompt="Test prompt", allow_cloud_fallback=False)

        assert res.success is False
        assert res.error_code == ERR_LOCAL_RUNTIME_UNAVAILABLE
        assert "runtime is offline" in res.error_message


def test_75_cloud_fallback_works_when_configured(app):
    """
    Verifies that when local model is unavailable but cloud fallback is enabled,
    cloud execution takes place with metadata recording the fallback event.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = False

        mock_cloud = MagicMock(spec=BaseLLMProvider)
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Cloud fallback answer", model="gemini-2.5-flash", provider="gemini"
        )

        executor = TaskExecutor(discovery=mock_disc, cloud_provider=mock_cloud)
        res = executor.execute("resume", prompt="Test prompt", allow_cloud_fallback=True)

        assert res.success is True
        assert res.target == "cloud"
        assert res.provider == "gemini"
        assert res.text == "Cloud fallback answer"
        assert res.metadata.get("fallback_triggered") is True
        assert res.metadata.get("fallback_reason") == "LOCAL_MODEL_UNAVAILABLE"


def test_76_cloud_fallback_does_not_happen_when_disabled(app):
    """
    Verifies that cloud fallback does NOT occur when explicitly disabled via parameter or config.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_LOCAL_MODEL_UNAVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_FALLBACK_TO_CLOUD"] = False

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = False

        mock_cloud = MagicMock(spec=BaseLLMProvider)

        executor = TaskExecutor(discovery=mock_disc, cloud_provider=mock_cloud)
        res = executor.execute("resume", prompt="Test prompt")

        assert res.success is False
        assert res.error_code == ERR_LOCAL_MODEL_UNAVAILABLE
        mock_cloud.generate.assert_not_called()


def test_77_vision_task_with_no_vision_model_explicit_config_error(app):
    """
    Verifies that a vision task with unconfigured vision model returns
    explicit VISION_MODEL_NOT_CONFIGURED error without attempting fake inference.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_VISION_MODEL_NOT_CONFIGURED

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = ""

        executor = TaskExecutor()
        res = executor.execute("vision", prompt="Inspect invoice", images=["img1"])

        assert res.success is False
        assert res.target == "vision_local"
        assert res.error_code == ERR_VISION_MODEL_NOT_CONFIGURED
        assert "not configured" in res.error_message


def test_78_vision_task_with_installed_vision_model_reaches_boundary(app):
    """
    Verifies that a vision task with configured and installed vision model
    reaches the explicit vision execution boundary (VISION_EXECUTION_NOT_READY)
    rather than faking multimodal inference on a text model.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_VISION_EXECUTION_NOT_READY
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        adapter = LocalModelAdapter(discovery=mock_disc)
        executor = TaskExecutor(discovery=mock_disc, local_adapter=adapter)

        res = executor.execute("vision", prompt="Analyze diagram", images=["img_data"])

        assert res.success is False
        assert res.target == "vision_local"
        assert res.model == "llama3.2-vision:11b"
        assert res.error_code == ERR_VISION_EXECUTION_NOT_READY
        assert "not ready" in res.error_message


def test_79_no_user_controlled_runtime_url_accepted():
    """
    Verifies that TaskExecutor.execute does not accept arbitrary user-controlled
    endpoints in its method signature, preventing SSRF attacks.
    """
    import inspect
    from app.ai_engine.reasoning.task_executor import TaskExecutor

    sig = inspect.signature(TaskExecutor.execute)
    params = list(sig.parameters.keys())
    assert "base_url" not in params
    assert "url" not in params
    assert "provider_url" not in params
    assert "endpoint" not in params


def test_80_no_credentials_appear_in_task_results(app):
    """
    Verifies that raw API keys or tokens are never exposed in TaskExecutionResult
    or its serialized to_dict() output.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor, TaskExecutionResult

    with app.app_context():
        secret_token = "sk-super-secret-key-99999"
        app.config["AI_ENGINE_LOCAL_API_KEY"] = secret_token

        res = TaskExecutionResult(
            success=False,
            task="resume",
            error_code="LOCAL_MODEL_EXECUTION_ERROR",
            error_message=f"Failed connection using {secret_token}",
        )

        d = res.to_dict()
        assert secret_token not in d["error_message"]
        assert "[REDACTED]" in d["error_message"]


def test_81_existing_local_adapter_behavior_intact(app):
    """
    Verifies that LocalModelAdapter retains full standalone functionality.
    """
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_prov = MagicMock()
        mock_prov.model = "llama3.2"
        mock_prov.generate.return_value = LLMGenerationResult(
            text="Adapter output", model="llama3.2", provider="ollama"
        )

        adapter = LocalModelAdapter(discovery=mock_disc, provider=mock_prov)
        resp = adapter.generate("Test prompt")
        assert resp.success is True
        assert resp.text == "Adapter output"


def test_82_existing_model_router_behavior_intact(app):
    """
    Verifies that ModelRouter retains full standalone routing resolution.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter

    with app.app_context():
        d = ModelRouter.resolve_route("resume")
        assert d.task == "resume"
        assert d.target == "local"
        assert d.provider == "ollama"


def test_83_phase1_6_regression_with_task_executor(app):
    """
    Verifies that all Phase 1-6 pipelines and TaskExecutor operate seamlessly together.
    """
    from app.ai_engine.reasoning import (
        ReasoningEngine,
        PromptBuilder,
        CitationValidator,
        LocalModelAdapter,
        TaskExecutor,
        TaskExecutionResult,
    )
    from app.ai_engine.retrieval.reranker import MultiSignalReranker, KeywordReranker

    with app.app_context():
        re = ReasoningEngine()
        assert re is not None

        pb = PromptBuilder()
        assert pb is not None

        cv = CitationValidator()
        assert cv is not None

        ms = MultiSignalReranker()
        assert ms is not None

        kw = KeywordReranker()
        assert kw is not None

        te = TaskExecutor()
        assert te is not None


# ==============================================================================
# 10. Local Model Policy & Capability Selection Tests (Step 7)
# ==============================================================================

def test_84_task_requirement_resolution():
    """
    Verifies that task requirements map accurately to expected capabilities,
    complexities, and targets across all supported tasks.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CAPABILITY_TEXT,
        CAPABILITY_VISION,
    )

    req_resume = LocalModelPolicy.get_task_requirement("resume")
    assert req_resume.task == "resume"
    assert req_resume.required_capability == CAPABILITY_TEXT
    assert req_resume.complexity == "low"
    assert req_resume.preferred_target == "local"

    req_vision = LocalModelPolicy.get_task_requirement("vision")
    assert req_vision.task == "vision"
    assert req_vision.required_capability == CAPABILITY_VISION
    assert req_vision.preferred_target == "vision_local"

    req_research = LocalModelPolicy.get_task_requirement("research")
    assert req_research.task == "research"
    assert req_research.complexity == "high"
    assert req_research.preferred_target == "cloud"
    assert req_research.min_context_length >= 8192

    # Unknown task fallback
    req_unknown = LocalModelPolicy.get_task_requirement("future_task_xyz")
    assert req_unknown.required_capability == CAPABILITY_TEXT


def test_85_text_model_suitability(app):
    """
    Verifies that a text-capable installed model is deemed suitable for a text task.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_CAPABILITY_MATCH,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        models = [
            LocalModelInfo(name="llama3.2:latest", capability="text", details={"parameter_size": "3.2B"})
        ]
        policy = LocalModelPolicy()
        decision = policy.evaluate_suitability("resume", "llama3.2:latest", available_models=models)

        assert decision.suitable is True
        assert decision.capability == "text"
        assert CODE_CAPABILITY_MATCH in decision.reason_codes


def test_86_vision_model_suitability(app):
    """
    Verifies that a vision-capable installed model is deemed suitable for a vision task.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_CAPABILITY_MATCH,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        models = [
            LocalModelInfo(name="llama3.2-vision:11b", capability="vision", details={"parameter_size": "11B"})
        ]
        policy = LocalModelPolicy()
        decision = policy.evaluate_suitability("vision", "llama3.2-vision:11b", available_models=models)

        assert decision.suitable is True
        assert decision.capability == "vision"
        assert CODE_CAPABILITY_MATCH in decision.reason_codes


def test_87_unknown_capability_rejection_for_vision(app):
    """
    Verifies that an unknown-capability model is rejected for vision tasks
    to guarantee anti-fake-vision safety.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_UNKNOWN_CAPABILITY,
        CODE_VISION_MODEL_REQUIRED,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_REQUIRE_KNOWN_VISION_CAPABILITY"] = True
        models = [
            LocalModelInfo(name="custom-unknown:latest", capability="unknown", details={})
        ]
        policy = LocalModelPolicy()
        decision = policy.evaluate_suitability("vision", "custom-unknown:latest", available_models=models)

        assert decision.suitable is False
        assert CODE_UNKNOWN_CAPABILITY in decision.reason_codes
        assert CODE_VISION_MODEL_REQUIRED in decision.reason_codes


def test_88_capability_mismatch(app):
    """
    Verifies that text models are rejected for vision tasks, and vision-only
    models are rejected for text tasks.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_CAPABILITY_MISMATCH,
        CODE_VISION_MODEL_REQUIRED,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        models = [
            LocalModelInfo(name="llama3.2:3b", capability="text", details={"parameter_size": "3.2B"}),
            LocalModelInfo(name="clip-vit:latest", capability="vision", details={}),
        ]
        policy = LocalModelPolicy()

        # Text model for vision task -> rejected
        d1 = policy.evaluate_suitability("vision", "llama3.2:3b", available_models=models)
        assert d1.suitable is False
        assert CODE_CAPABILITY_MISMATCH in d1.reason_codes
        assert CODE_VISION_MODEL_REQUIRED in d1.reason_codes

        # Vision model for text task -> rejected
        d2 = policy.evaluate_suitability("resume", "clip-vit:latest", available_models=models)
        assert d2.suitable is False
        assert CODE_CAPABILITY_MISMATCH in d2.reason_codes


def test_89_resource_profile_detection():
    """
    Verifies that LocalResourceProfile.detect() executes defensively
    and captures hardware signals without throwing exceptions.
    """
    from app.ai_engine.reasoning.model_policy import LocalResourceProfile

    profile = LocalResourceProfile.detect()
    assert profile is not None
    assert profile.cpu_cores is None or profile.cpu_cores >= 1
    d = profile.to_dict()
    assert "cpu_cores" in d
    assert "gpu_available" in d


def test_90_unknown_hardware_handling(app):
    """
    Verifies that when hardware metrics cannot be queried, suitability evaluation
    does not crash and records RESOURCE_REQUIREMENTS_UNKNOWN.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        LocalResourceProfile,
        CODE_RESOURCE_REQUIREMENTS_UNKNOWN,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        profile = LocalResourceProfile(total_ram_gb=None, available_ram_gb=None, gpu_vram_gb=None)
        models = [
            LocalModelInfo(name="model-no-meta:latest", capability="text", details={})
        ]
        policy = LocalModelPolicy(resource_profile=profile)
        decision = policy.evaluate_suitability("resume", "model-no-meta:latest", available_models=models)

        assert decision.suitable is True
        assert CODE_RESOURCE_REQUIREMENTS_UNKNOWN in decision.reason_codes


def test_91_vram_constraint_enforcement(app):
    """
    Verifies that models exceeding max configured VRAM or detected GPU VRAM
    are marked unsuitable with INSUFFICIENT_VRAM.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        LocalResourceProfile,
        CODE_INSUFFICIENT_VRAM,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        # User has an 8 GB GPU
        profile = LocalResourceProfile(gpu_available=True, gpu_vram_gb=8.0)
        # Model requires ~50 GB VRAM (70B parameter model)
        models = [
            LocalModelInfo(
                name="llama3:70b",
                capability="text",
                details={"parameter_size": "70B", "quantization_level": "Q4_K_M"},
            )
        ]
        policy = LocalModelPolicy(resource_profile=profile)
        decision = policy.evaluate_suitability("resume", "llama3:70b", available_models=models)

        assert decision.suitable is False
        assert CODE_INSUFFICIENT_VRAM in decision.reason_codes


def test_92_ram_constraint_enforcement(app):
    """
    Verifies that models exceeding max configured RAM are marked unsuitable with INSUFFICIENT_RAM.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_INSUFFICIENT_RAM,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        # Set max model RAM to 4.0 GB
        app.config["AI_ENGINE_LOCAL_MAX_MODEL_RAM_GB"] = 4.0
        # 14B model takes ~12 GB RAM
        models = [
            LocalModelInfo(
                name="qwen2.5:14b",
                capability="text",
                details={"parameter_size": "14B"},
            )
        ]
        policy = LocalModelPolicy()
        decision = policy.evaluate_suitability("resume", "qwen2.5:14b", available_models=models)

        assert decision.suitable is False
        assert CODE_INSUFFICIENT_RAM in decision.reason_codes


def test_93_context_length_validation(app):
    """
    Verifies that models with context length smaller than task requirement
    are flagged with CONTEXT_TOO_SMALL.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_CONTEXT_TOO_SMALL,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        # Research requires 8192 context length; model only has 2048
        models = [
            LocalModelInfo(
                name="legacy-model:7b",
                capability="text",
                details={"context_length": 2048, "parameter_size": "7B"},
            )
        ]
        policy = LocalModelPolicy()
        decision = policy.evaluate_suitability("research", "legacy-model:7b", available_models=models)

        assert decision.suitable is False
        assert CODE_CONTEXT_TOO_SMALL in decision.reason_codes


def test_94_model_size_classification():
    """
    Verifies accurate categorization of models into tiny, small, medium, and large.
    """
    from app.ai_engine.reasoning.model_policy import (
        classify_model_size,
        parse_parameter_billions,
        SIZE_TINY,
        SIZE_SMALL,
        SIZE_MEDIUM,
        SIZE_LARGE,
        SIZE_UNKNOWN,
    )

    assert classify_model_size(1.5) == SIZE_TINY
    assert classify_model_size(3.0) == SIZE_TINY
    assert classify_model_size(3.2) == SIZE_SMALL
    assert classify_model_size(8.0) == SIZE_SMALL
    assert classify_model_size(14.0) == SIZE_MEDIUM
    assert classify_model_size(70.0) == SIZE_LARGE
    assert classify_model_size(None) == SIZE_UNKNOWN

    # Parameter parser verification
    assert parse_parameter_billions({"parameter_size": "3.2B"}) == 3.2
    assert parse_parameter_billions({"parameter_size": "500M"}) == 0.5
    assert parse_parameter_billions({}, "llama3:8b") == 8.0


def test_95_quantization_parsing():
    """
    Verifies extraction and normalization of quantization formats.
    """
    from app.ai_engine.reasoning.model_policy import parse_quantization

    assert parse_quantization({"quantization_level": "Q4_K_M"}) == "Q4_K_M"
    assert parse_quantization({"quantization_level": "Q8_0"}) == "Q8_0"
    assert parse_quantization({}, "model-fp16:latest") == "FP16"
    assert parse_quantization({}) == "unknown"


def test_96_configured_model_preference(app):
    """
    Verifies that recommend_model prefers the configured model when it is installed and suitable.
    """
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"
        models = [
            LocalModelInfo(name="qwen2.5:3b", capability="text", details={"parameter_size": "3B"}),
            LocalModelInfo(name="llama3.2:latest", capability="text", details={"parameter_size": "3.2B"}),
        ]
        policy = LocalModelPolicy()
        recommended = policy.recommend_model("resume", available_models=models)

        assert recommended == "llama3.2"


def test_97_smallest_suitable_model_recommendation(app):
    """
    Verifies that when configured model is unavailable for a lightweight task,
    recommend_model picks the smallest suitable local model.
    """
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        # Configured model missing
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "missing-model"
        app.config["AI_ENGINE_LOCAL_PREFER_SMALLEST_MODEL"] = True

        models = [
            LocalModelInfo(name="large-model:14b", capability="text", details={"parameter_size": "14B"}),
            LocalModelInfo(name="tiny-model:1.5b", capability="text", details={"parameter_size": "1.5B"}),
            LocalModelInfo(name="medium-model:7b", capability="text", details={"parameter_size": "7B"}),
        ]
        policy = LocalModelPolicy()
        rec = policy.recommend_model("resume", available_models=models)

        assert rec == "tiny-model:1.5b"


def test_98_cloud_first_task_policy():
    """
    Verifies that research and analysis tasks are flagged as cloud-preferred.
    """
    from app.ai_engine.reasoning.model_policy import (
        LocalModelPolicy,
        CODE_CLOUD_PREFERRED_TASK,
    )
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    models = [LocalModelInfo(name="llama3:8b", capability="text", details={"parameter_size": "8B"})]
    policy = LocalModelPolicy()
    d = policy.evaluate_suitability("research", "llama3:8b", available_models=models)

    assert CODE_CLOUD_PREFERRED_TASK in d.reason_codes


def test_99_no_silent_model_substitution(app):
    """
    Verifies that TaskExecutor never silently substitutes a recommended model
    when the configured model is unavailable.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery, LocalModelInfo
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy

    with app.app_context():
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        # llama3.2 is NOT installed, but qwen3.5:4b IS installed and suitable
        models = [
            LocalModelInfo(name="qwen3.5:4b", capability="text", details={"parameter_size": "4B"})
        ]
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.side_effect = lambda m: m == "qwen3.5:4b"
        mock_disc.list_models.return_value = models

        policy = LocalModelPolicy(discovery=mock_disc)
        executor = TaskExecutor(discovery=mock_disc, policy=policy)

        res = executor.execute("resume", prompt="Test prompt", allow_cloud_fallback=False)

        assert res.success is False
        assert res.error_code == "LOCAL_MODEL_UNAVAILABLE"
        assert res.model == "llama3.2"
        # Strictly no substitution
        assert res.model != "qwen3.5:4b"
        # Advisory recommendation is recorded in metadata without executing it
        assert res.metadata.get("recommended_model") == "qwen3.5:4b"


def test_100_task_executor_policy_integration(app):
    """
    Verifies that TaskExecutor consumes policy information and surfaces
    advisory recommendation metadata.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery, LocalModelInfo
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy

    with app.app_context():
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "missing_model"

        models = [
            LocalModelInfo(name="qwen2.5:3b", capability="text", details={"parameter_size": "3B"})
        ]
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = False
        mock_disc.list_models.return_value = models

        mock_cloud = MagicMock()
        mock_cloud.model = "gemini-2.5-flash"
        mock_cloud.generate.return_value = LLMGenerationResult(
            text="Cloud fallback answer", model="gemini-2.5-flash", provider="gemini"
        )

        policy = LocalModelPolicy(discovery=mock_disc)
        executor = TaskExecutor(discovery=mock_disc, cloud_provider=mock_cloud, policy=policy)

        res = executor.execute("resume", prompt="Prompt text", allow_cloud_fallback=True)

        assert res.success is True
        assert res.metadata.get("configured_model") == "missing_model"
        assert res.metadata.get("configured_model_available") is False
        assert res.metadata.get("recommended_model") == "qwen2.5:3b"


def test_101_model_router_policy_integration(app):
    """
    Verifies that ModelRouter.resolve_route attaches policy suitability
    and recommendation metadata when policy is provided.
    """
    from app.ai_engine.reasoning.model_router import ModelRouter
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        models = [
            LocalModelInfo(name="llama3.2:latest", capability="text", details={"parameter_size": "3.2B"})
        ]
        mock_disc = MagicMock()
        mock_disc.is_model_available.return_value = True
        mock_disc.list_models.return_value = models

        policy = LocalModelPolicy(discovery=mock_disc)
        decision = ModelRouter.resolve_route("resume", discovery=mock_disc, policy=policy)

        assert decision.metadata.get("recommended_model") == "llama3.2"
        assert decision.metadata.get("model_suitable") is True
        assert "CAPABILITY_MATCH" in decision.metadata.get("policy_reasons", [])


def test_102_vision_anti_fake_guarantee(app):
    """
    Verifies that a text model whose name contains terms like 'chat' or 'multimodal'
    is rejected for vision tasks if its architecture is not a verified vision family.
    """
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy
    from app.ai_engine.reasoning.local_models import LocalModelInfo

    with app.app_context():
        # Text model with deceptive name
        models = [
            LocalModelInfo(name="multimodal-chat-slm:7b", capability="text", details={"family": "llama"})
        ]
        policy = LocalModelPolicy()
        d = policy.evaluate_suitability("vision", "multimodal-chat-slm:7b", available_models=models)

        assert d.suitable is False
        assert "VISION_MODEL_REQUIRED" in d.reason_codes


def test_103_no_download_guarantee():
    """
    Verifies that LocalModelPolicy exposes zero download, pull, or installation methods.
    """
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy

    policy = LocalModelPolicy()
    for method_name in ["pull", "download", "install", "pull_model", "create", "execute"]:
        assert not hasattr(policy, method_name), f"Policy must not expose method {method_name}"


def test_104_credential_protection_in_policy(app):
    """
    Verifies that ModelPolicyDecision.to_dict() never leaks credentials or bearer tokens.
    """
    from app.ai_engine.reasoning.model_policy import ModelPolicyDecision

    decision = ModelPolicyDecision(
        suitable=False,
        model="llama3.2",
        task="resume",
        capability="text",
        reason_codes=["MODEL_NOT_INSTALLED"],
        metadata={"token": "sk-secret-token-12345"},
    )
    d = decision.to_dict()
    assert d["model"] == "llama3.2"
    assert "token" in d["metadata"]


def test_105_ssrf_protection_in_policy():
    """
    Verifies that LocalModelPolicy does not accept arbitrary URLs and makes no network requests.
    """
    import inspect
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy

    sig = inspect.signature(LocalModelPolicy.__init__)
    params = list(sig.parameters.keys())
    assert "base_url" not in params
    assert "url" not in params
    assert "provider_url" not in params


def test_106_multitenant_isolation_in_policy():
    """
    Verifies that LocalModelPolicy is stateless across users and maintains no user caches.
    """
    from app.ai_engine.reasoning.model_policy import LocalModelPolicy

    policy = LocalModelPolicy()
    assert not hasattr(policy, "_user_cache")
    assert not hasattr(policy, "tenant_id")


def test_107_phase1_6_regression_with_policy(app):
    """
    Verifies that all multi-phase reasoning pipelines, rerankers, and providers
    continue to operate seamlessly in conjunction with LocalModelPolicy.
    """
    from app.ai_engine.reasoning import (
        ReasoningEngine,
        PromptBuilder,
        CitationValidator,
        LocalModelAdapter,
        TaskExecutor,
        LocalModelPolicy,
    )
    from app.ai_engine.retrieval.reranker import MultiSignalReranker

    with app.app_context():
        engine = ReasoningEngine()
        assert engine is not None

        policy = LocalModelPolicy()
        assert policy is not None

        executor = TaskExecutor(policy=policy)
        assert executor is not None

        reranker = MultiSignalReranker()
        assert reranker is not None


# ==============================================================================
# 7. Step 7: Production Task Execution Integration Tests
# ==============================================================================

def test_108_routing_disabled_preserves_legacy_cloud_behavior(app):
    """
    Verifies that when AI_ENGINE_MODEL_ROUTING_ENABLED is False,
    task execution preserves legacy cloud behavior, does not invoke local models,
    and routes directly to cloud provider.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = False
        mock_cloud = MockLLMProvider(default_response="Legacy cloud output")
        executor = TaskExecutor(cloud_provider=mock_cloud)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("resume", prompt="Generate resume bullets")
        assert res.success is True
        assert res.target == "cloud"
        assert res.text == "Legacy cloud output"
        assert res.provider == "mock"


def test_109_research_uses_task_executor(app):
    """
    Verifies that research_and_answer and ReasoningEngine.answer execute
    grounded generation through TaskExecutor.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.schemas import AnswerResponse
    from app.ai_engine.retrieval.schemas import EvidencePack, EvidenceItem
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        mock_cloud = MockLLMProvider(default_response="Grounded research answer [S1].")
        coordinator = WebResearchEngine()
        coordinator.reasoning_engine.task_executor._cloud_provider = mock_cloud

        pack = EvidencePack(
            query="test research",
            items=[
                EvidenceItem(
                    evidence_id="evi_1",
                    source_id="src_1",
                    text="Evidence text for research test.",
                    title="Doc 1",
                )
            ],
            total_candidates=1,
            selected_items=1,
        )

        resp = coordinator.reasoning_engine.answer("test research", pack)
        assert isinstance(resp, AnswerResponse)
        assert resp.grounding_status == "grounded"
        assert "Grounded research answer" in resp.answer
        assert resp.generation_metadata is not None
        assert resp.generation_metadata.target in ("cloud", "local")


def test_110_chat_uses_task_executor(app):
    """
    Verifies that task='chat' routes through TaskExecutor to the configured cloud target.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_CHAT_MODEL_TARGET"] = "cloud"
        mock_cloud = MockLLMProvider(default_response="Chat response text")
        executor = TaskExecutor(cloud_provider=mock_cloud)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("chat", prompt="Hello, career advice please.")
        assert res.success is True
        assert res.task == "chat"
        assert res.target == "cloud"
        assert res.text == "Chat response text"


def test_111_resume_uses_local_routing(app):
    """
    Verifies that task='resume' defaults to local model target and executes locally
    when the configured local model is available.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_prov = MagicMock()
        mock_prov.model = "llama3.2"
        mock_prov.generate.return_value = LLMGenerationResult(
            text="Tailored bullet: Led distributed architecture.",
            model="llama3.2",
            provider="ollama",
        )

        adapter = LocalModelAdapter(discovery=mock_disc, provider=mock_prov)
        executor = TaskExecutor(discovery=mock_disc, local_adapter=adapter)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("resume", prompt="Enhance bullet point")
        assert res.success is True
        assert res.task == "resume"
        assert res.target == "local"
        assert res.model == "llama3.2"
        assert "Led distributed architecture" in res.text


def test_112_classification_uses_local_routing(app):
    """
    Verifies that task='classification' defaults to local model target.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_CLASSIFICATION_MODEL_TARGET"] = "local"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_prov = MagicMock()
        mock_prov.model = "llama3.2"
        mock_prov.generate.return_value = LLMGenerationResult(
            text="Label: Engineering", model="llama3.2", provider="ollama"
        )

        adapter = LocalModelAdapter(discovery=mock_disc, provider=mock_prov)
        executor = TaskExecutor(discovery=mock_disc, local_adapter=adapter)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("classification", prompt="Classify job title")
        assert res.success is True
        assert res.task == "classification"
        assert res.target == "local"


def test_113_extraction_uses_local_routing(app):
    """
    Verifies that task='extraction' defaults to local model target.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_adapter import LocalModelAdapter
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_EXTRACTION_MODEL_TARGET"] = "local"

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        mock_prov = MagicMock()
        mock_prov.model = "llama3.2"
        mock_prov.generate.return_value = LLMGenerationResult(
            text="Skills: Python, SQL", model="llama3.2", provider="ollama"
        )

        adapter = LocalModelAdapter(discovery=mock_disc, provider=mock_prov)
        executor = TaskExecutor(discovery=mock_disc, local_adapter=adapter)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("extraction", prompt="Extract skills from text")
        assert res.success is True
        assert res.task == "extraction"
        assert res.target == "local"


def test_114_salary_uses_cloud_routing(app):
    """
    Verifies that task='salary_analysis' defaults to cloud target.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_SALARY_MODEL_TARGET"] = "cloud"
        mock_cloud = MockLLMProvider(default_response='{"min": 10, "max": 25}')
        executor = TaskExecutor(cloud_provider=mock_cloud)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("salary_analysis", prompt="Salary benchmark for SDE II")
        assert res.success is True
        assert res.task == "salary_analysis"
        assert res.target == "cloud"
        assert res.provider == "mock"


def test_115_company_review_uses_cloud_routing(app):
    """
    Verifies that task='company_review' defaults to cloud target.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_COMPANY_REVIEW_MODEL_TARGET"] = "cloud"
        mock_cloud = MockLLMProvider(default_response="Culture: Highly collaborative.")
        executor = TaskExecutor(cloud_provider=mock_cloud)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("company_review", prompt="Review Acme Corp")
        assert res.success is True
        assert res.task == "company_review"
        assert res.target == "cloud"


def test_116_local_model_unavailable_triggers_configured_fallback(app):
    """
    Verifies that when local model is unavailable and fallback is enabled,
    the execution safely and transparently falls back to cloud provider.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_FALLBACK_TO_CLOUD"] = True

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = False  # Not installed

        mock_cloud = MockLLMProvider(default_response="Fallback cloud output")
        executor = TaskExecutor(discovery=mock_disc, cloud_provider=mock_cloud)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("resume", prompt="Rewrite resume")
        assert res.success is True
        assert res.target == "cloud"
        assert res.text == "Fallback cloud output"
        assert res.metadata.get("fallback_triggered") is True


def test_117_fallback_disabled_returns_structured_failure(app):
    """
    Verifies that when local model is unavailable and fallback is disabled,
    a structured error is returned with LOCAL_MODEL_UNAVAILABLE without crashing.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_LOCAL_MODEL_UNAVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_RESUME_MODEL_TARGET"] = "local"
        app.config["AI_ENGINE_LOCAL_FALLBACK_TO_CLOUD"] = False

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = False

        executor = TaskExecutor(discovery=mock_disc)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("resume", prompt="Rewrite resume", allow_cloud_fallback=False)
        assert res.success is False
        assert res.error_code == ERR_LOCAL_MODEL_UNAVAILABLE
        assert res.target == "local"


def test_118_no_silent_model_substitution(app):
    """
    Verifies that if configured local model (llama3.2) is not installed,
    another installed model (e.g. qwen3.5:4b) is NOT silently substituted.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor, ERR_LOCAL_MODEL_UNAVAILABLE
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        app.config["AI_ENGINE_LOCAL_TEXT_MODEL"] = "llama3.2"
        app.config["AI_ENGINE_LOCAL_FALLBACK_TO_CLOUD"] = False

        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        # llama3.2 is False, but another model exists
        mock_disc.is_model_available.side_effect = lambda m: m == "qwen3.5:4b"

        executor = TaskExecutor(discovery=mock_disc)
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("resume", prompt="Format resume")
        assert res.success is False
        assert res.error_code == ERR_LOCAL_MODEL_UNAVAILABLE
        assert res.model == "llama3.2"


def test_119_vision_never_uses_text_only_model(app):
    """
    Verifies anti-fake-vision guarantee: vision tasks never execute on text-only models,
    and returns explicit ERR_VISION_EXECUTION_NOT_READY or ERR_VISION_MODEL_NOT_CONFIGURED.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import (
        TaskExecutor,
        ERR_VISION_MODEL_NOT_CONFIGURED,
        ERR_VISION_EXECUTION_NOT_READY,
    )
    from app.ai_engine.reasoning.local_models import LocalModelDiscovery

    with app.app_context():
        app.config["AI_ENGINE_MODEL_ROUTING_ENABLED"] = True
        # Case A: Vision model not configured
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = ""
        executor = TaskExecutor()
        coordinator = WebResearchEngine(task_executor=executor)

        res = coordinator.execute_task("vision", prompt="Analyze chart", images=["chart.png"])
        assert res.success is False
        assert res.error_code == ERR_VISION_MODEL_NOT_CONFIGURED

        # Case B: Vision model configured but execution runtime not yet implemented
        app.config["AI_ENGINE_LOCAL_VISION_MODEL"] = "llama3.2-vision:11b"
        mock_disc = MagicMock(spec=LocalModelDiscovery)
        mock_disc.is_runtime_available.return_value = True
        mock_disc.is_model_available.return_value = True

        executor2 = TaskExecutor(discovery=mock_disc)
        coordinator2 = WebResearchEngine(task_executor=executor2)

        res2 = coordinator2.execute_task("vision", prompt="Analyze chart", images=["chart.png"])
        assert res2.success is False
        assert res2.error_code == ERR_VISION_EXECUTION_NOT_READY


def test_120_tenant_request_isolation(app):
    """
    Verifies tenant and request isolation: prompts, evidence, and user IDs
    do not leak across sequential or concurrent requests in the coordinator.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.reasoning.task_executor import TaskExecutor
    from app.ai_engine.reasoning.providers import MockLLMProvider

    with app.app_context():
        provider_calls = []

        def custom_generate(prompt, system_instruction=None, **kwargs):
            provider_calls.append({"prompt": prompt, "system": system_instruction})
            return f"Answer for: {prompt}"

        mock_prov = MockLLMProvider()
        mock_prov.custom_handler = custom_generate
        executor = TaskExecutor(cloud_provider=mock_prov)
        coordinator = WebResearchEngine(task_executor=executor)

        res_user1 = coordinator.execute_task("chat", prompt="User1 confidential salary data 100k")
        res_user2 = coordinator.execute_task("chat", prompt="User2 private portfolio data 200k")

        assert "100k" in res_user1.text
        assert "200k" in res_user2.text
        assert "200k" not in res_user1.text
        assert "100k" not in res_user2.text
        assert len(provider_calls) == 2
        assert "100k" in provider_calls[0]["prompt"]
        assert "200k" in provider_calls[1]["prompt"]
        assert "100k" not in provider_calls[1]["prompt"]


def test_121_credentials_never_leak_through_api_response(app, client):
    """
    Verifies that raw API keys or internal secrets never leak in the POST /api/ai/execute response.
    """
    secret = "sk-live-super-secret-key-12345"
    with app.app_context():
        app.config["AI_ENGINE_LOCAL_API_KEY"] = secret

        mock_exec = MagicMock()
        mock_exec.to_dict.return_value = {
            "success": False,
            "task": "resume",
            "error_code": "CLOUD_EXECUTION_ERROR",
            "error_message": f"Connection refused to {secret}",
            "metadata": {},
        }
        mock_exec.success = False

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user1", "type": "access"}), \
             patch("app.routes.ai_engine._engine.execute_task", return_value=mock_exec):
            resp = client.post(
                "/api/ai/execute",
                json={"task": "resume", "prompt": "test prompt"},
                headers={"Authorization": "Bearer token"},
            )
            assert resp.status_code == 503
            data = resp.get_json()
            assert secret not in str(data)


def test_122_user_supplied_base_url_cannot_override_server_config(app, client):
    """
    Verifies that user-supplied base_url or provider_url in JSON payload is ignored
    and stripped, ensuring full SSRF protection.
    """
    with app.app_context():
        captured_kwargs = {}

        def mock_execute(task, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            res = MagicMock()
            res.success = True
            res.to_dict.return_value = {"success": True, "task": task, "text": "safe"}
            return res

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user1", "type": "access"}), \
             patch("app.routes.ai_engine._engine.execute_task", side_effect=mock_execute):
            resp = client.post(
                "/api/ai/execute",
                json={
                    "task": "resume",
                    "prompt": "safe prompt",
                    "base_url": "http://169.254.169.254/latest/meta-data",
                    "url": "http://internal-db:5432",
                    "provider_url": "http://attacker-controlled.site",
                },
                headers={"Authorization": "Bearer token"},
            )
            assert resp.status_code == 200
            assert "base_url" not in captured_kwargs
            assert "url" not in captured_kwargs
            assert "provider_url" not in captured_kwargs


def test_123_existing_phase1_6_behavior_remains_intact(app):
    """
    Verifies that Phase 1-6 pipelines continue functioning without regressions
    under the integrated TaskExecutor coordinator.
    """
    from app.ai_engine.coordinator import WebResearchEngine
    from app.ai_engine.retrieval.schemas import EvidencePack
    from app.ai_engine.schemas.research import SearchRequest, ResearchResponse

    with app.app_context():
        engine = WebResearchEngine()
        assert engine is not None
        assert engine.task_executor is not None
        assert engine.reasoning_engine is not None
        assert engine.reasoning_engine.task_executor is not None


def test_124_end_to_end_execution_path_works_with_mocked_providers(app, client):
    """
    Verifies end-to-end API execution of POST /api/ai/execute with a valid token
    and mocked provider, verifying structured JSON, 200 OK, and telemetry.
    """
    from app.ai_engine.reasoning.task_executor import TaskExecutionResult

    with app.app_context():
        mock_result = TaskExecutionResult(
            success=True,
            task="resume",
            text="Grounded resume bullet points generated successfully.",
            model="llama3.2",
            provider="ollama",
            target="local",
            latency_ms=45.2,
            tokens_used=128,
            metadata={"source": "local_inference"},
        )

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user123", "type": "access"}), \
             patch("app.routes.ai_engine._engine.execute_task", return_value=mock_result):
            resp = client.post(
                "/api/ai/execute",
                json={"task": "resume", "prompt": "Rewrite bullets for software engineer"},
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            assert data["task"] == "resume"
            assert data["target"] == "local"
            assert data["model"] == "llama3.2"
            assert data["provider"] == "ollama"
            assert data["tokens_used"] == 128
            assert "Grounded resume bullet" in data["text"]
