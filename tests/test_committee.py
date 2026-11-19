import time

import pandas as pd
import pytest

from goldbot.config import JudgeConfig
from goldbot.judge import Committee
from goldbot.judge.providers import (Provider, ProviderError, available_providers, make_provider,
                                     parse_json_text, register)
from goldbot.models import Setup

T = pd.Timestamp("2026-03-10 13:20")


def make_setup(d=1, trend=1):
    return Setup("orb", "g", d, 2000.0, 1995.0, 2015.0, T, T, T, context={"trend_d1": trend})


class Scripted(Provider):
    """Returns votes from a list, one per call (thread-safe enough for 4 calls)."""
    default_model = "scripted"

    def __init__(self, votes, delay=0.0, **kw):
        super().__init__(**kw)
        self.votes, self.delay, self.calls = list(votes), delay, 0

    def complete_json(self, system, user, schema):
        time.sleep(self.delay)
        self.calls += 1
        v = self.votes.pop(0)
        if isinstance(v, Exception):
            raise v
        return {"approve": v, "confidence": 0.8, "reason": "scripted"}


def cfg(**kw):
    return JudgeConfig(enabled=True, provider="fake", **kw)


def test_builtin_providers_registered():
    assert {"anthropic", "openai", "openai_compatible", "fake"} <= available_providers()


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        make_provider("nope")


def test_openai_requires_model():
    with pytest.raises((ValueError, ProviderError)):
        make_provider("openai", api_key="x")


def test_custom_provider_plugs_in():
    @register("unit_test_provider")
    class P(Provider):
        default_model = "m"

        def complete_json(self, system, user, schema):
            return {"approve": True, "confidence": 1, "reason": "ok"}

    c = Committee(cfg(use_chair=False), provider=make_provider("unit_test_provider"))
    assert c.evaluate(make_setup(), {}, {}).approved


def test_majority_then_chair():
    c = Committee(cfg(), provider=Scripted([True, True, False, True]))
    d = c.evaluate(make_setup(), {}, {})
    assert d.approved and set(d.votes) == {"trend_analyst", "volatility_analyst", "risk_officer", "chair"}


def test_chair_can_veto():
    d = Committee(cfg(), provider=Scripted([True, True, True, False])).evaluate(make_setup(), {}, {})
    assert not d.approved


def test_not_enough_analysts_skips_chair():
    p = Scripted([True, False, False])
    d = Committee(cfg(), provider=p).evaluate(make_setup(), {}, {})
    assert not d.approved and p.calls == 3


def test_fail_open_and_fail_closed():
    err = [ProviderError("down")] * 4
    assert Committee(cfg(fail_mode="open"), provider=Scripted(err)).evaluate(make_setup(), {}, {}).approved
    d = Committee(cfg(fail_mode="closed"), provider=Scripted(list(err))).evaluate(make_setup(), {}, {})
    assert not d.approved and d.error


def test_timeout_triggers_fail_mode_quickly():
    p = Scripted([True] * 4, delay=2.0)
    t0 = time.monotonic()
    d = Committee(cfg(fail_mode="closed", timeout_s=0.3), provider=p).evaluate(make_setup(), {}, {})
    assert not d.approved and time.monotonic() - t0 < 1.5


def test_parse_json_text_handles_fences():
    assert parse_json_text('```json\n{"approve": true}\n```') == {"approve": True}
    with pytest.raises(ProviderError):
        parse_json_text("no json here")


def test_fake_provider_end_to_end():
    c = Committee(cfg(), provider=make_provider("fake"))
    assert c.evaluate(make_setup(d=1, trend=1), {}, {}).approved
    assert not c.evaluate(make_setup(d=1, trend=-1), {}, {}).approved
