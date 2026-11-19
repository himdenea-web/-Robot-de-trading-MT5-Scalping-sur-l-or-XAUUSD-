"""Pluggable LLM provider slot.

One provider (selected in the config) serves every agent of the committee. A provider is a
class with a single method, `complete_json(system, user, schema) -> dict`, registered under a
name with @register("name"). Built in:

  anthropic          Claude through the official Anthropic SDK      (pip install "anthropic>=1.0")
  openai             OpenAI through the official OpenAI SDK         (pip install openai)
  openai_compatible  any OpenAI-compatible endpoint via GB_LLM_BASE_URL
                     (OpenRouter, Groq, Together, a local Ollama server, ...)

Adding another provider:

    from goldbot.judge.providers import Provider, register

    @register("myprovider")
    class MyProvider(Provider):
        def complete_json(self, system, user, schema):
            ...call your API, return a dict matching `schema`...

then set GB_LLM_PROVIDER=myprovider (import the module before the bot starts, e.g. in a
small launcher script).
"""
import json
import re

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.provider_name = name
        return cls
    return deco


def available_providers() -> set[str]:
    return set(_REGISTRY)


def make_provider(name: str, api_key: str = "", model: str = "", base_url: str = "",
                  timeout_s: float = 45.0):
    if name not in _REGISTRY:
        raise ValueError(f"unknown LLM provider {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](api_key=api_key, model=model, base_url=base_url, timeout_s=timeout_s)


class ProviderError(RuntimeError):
    """Raised for any provider failure; the committee turns it into its fail mode."""


class Provider:
    provider_name = "base"
    default_model = ""

    def __init__(self, api_key="", model="", base_url="", timeout_s=45.0):
        self.api_key = api_key
        self.model = model or self.default_model
        self.base_url = base_url
        self.timeout_s = timeout_s
        if not self.model:
            raise ValueError(f"provider {self.provider_name!r} needs GB_LLM_MODEL to be set")

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.provider_name}:{self.model}"


def parse_json_text(text: str) -> dict:
    """Parse a JSON object from model text, tolerating code fences around it."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ProviderError(f"no JSON object in model output: {text[:120]!r}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ProviderError(f"invalid JSON from model: {e}") from e


@register("anthropic")
class AnthropicProvider(Provider):
    """Claude via the Anthropic SDK, with structured outputs (JSON schema) and the
    server-side refusal fallback. Credentials: GB_LLM_API_KEY, or the SDK's usual
    resolution (ANTHROPIC_API_KEY / `ant auth login` profile) when left empty."""
    default_model = "claude-opus-5-5"

    def __init__(self, **kw):
        super().__init__(**kw)
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError('install the Anthropic SDK: pip install "anthropic>=1.0"') from e
        self._anthropic = anthropic
        kwargs = {"timeout": self.timeout_s, "max_retries": 2}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = anthropic.Anthropic(**kwargs)

    def complete_json(self, system, user, schema):
        a = self._anthropic
        try:
            resp = self._client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError as e:
            raise ProviderError("Anthropic: invalid API key") from e
        except a.RateLimitError as e:
            raise ProviderError("Anthropic: rate limited") from e
        except a.APIStatusError as e:
            raise ProviderError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise ProviderError("Anthropic: network error") from e
        if resp.stop_reason == "refusal":
            raise ProviderError("Anthropic: request declined")
        if resp.stop_reason == "max_tokens":
            raise ProviderError("Anthropic: output truncated")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_json_text(text)


@register("openai")
class OpenAIProvider(Provider):
    """OpenAI chat completions in JSON mode. GB_LLM_MODEL is required (no default)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        try:
            import openai
        except ImportError as e:
            raise ProviderError("install the OpenAI SDK: pip install openai") from e
        self._openai = openai
        kwargs = {"timeout": self.timeout_s, "max_retries": 2}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = openai.OpenAI(**kwargs)

    def complete_json(self, system, user, schema):
        o = self._openai
        prompt = (f"{user}\n\nReply with one JSON object that matches this JSON schema, "
                  f"and nothing else:\n{json.dumps(schema)}")
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except o.AuthenticationError as e:
            raise ProviderError(f"{self.provider_name}: invalid API key") from e
        except o.RateLimitError as e:
            raise ProviderError(f"{self.provider_name}: rate limited") from e
        except o.APIStatusError as e:
            raise ProviderError(f"{self.provider_name} API error {e.status_code}") from e
        except o.APIConnectionError as e:
            raise ProviderError(f"{self.provider_name}: network error") from e
        return parse_json_text(resp.choices[0].message.content or "")


@register("openai_compatible")
class OpenAICompatibleProvider(OpenAIProvider):
    """Any endpoint that speaks the OpenAI chat-completions API. Needs GB_LLM_BASE_URL
    and GB_LLM_MODEL (e.g. https://openrouter.ai/api/v1, http://localhost:11434/v1)."""

    def __init__(self, **kw):
        if not kw.get("base_url"):
            raise ValueError("openai_compatible needs GB_LLM_BASE_URL")
        super().__init__(**kw)


@register("fake")
class FakeProvider(Provider):
    """Deterministic offline provider for tests and dry runs: approves when the trend
    agrees with the trade direction. Never use for real decisions."""
    default_model = "fake"

    def complete_json(self, system, user, schema):
        long_ok = '"trend_d1": 1' in user
        short_ok = '"trend_d1": -1' in user
        side = "BUY" if '"side": "BUY"' in user else "SELL"
        ok = long_ok if side == "BUY" else short_ok
        return {"approve": ok, "confidence": 0.7 if ok else 0.6,
                "reason": "fake provider: trend agreement check"}
