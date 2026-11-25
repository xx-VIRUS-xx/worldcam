import json

import pytest

from worldcam.models.api import TASKS, GenRequest, NotConfigured
from worldcam.models.router import ConfiguredModelRouter, parse_spec
from worldcam.models.usage import UsageLog


def fake_gemini_transport(url, headers, body, timeout):
    return {"candidates": [{"content": {"parts": [{"text": "a red box"}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 100}}


def fake_spawn(cmd, stdin_text, env, cwd, timeout):
    return 0, json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "hi",
                          "usage": {"input_tokens": 5, "output_tokens": 2}, "total_cost_usd": 0.01}), ""


def router(config=None, env=None, **kw):
    return ConfiguredModelRouter(config=config or {}, env={} if env is None else env, **kw)


def test_defaults_cover_every_task():
    d = router().describe()
    assert set(d) == set(TASKS)
    assert d["vision_crop"].startswith("gemini:") and d["agent"].startswith("claude:")
    assert d["decide"] == "claude:haiku"


def test_decide_defaults_to_jev_only_when_a_key_is_present():
    assert router(env={"JEV_API_KEY": "k"}).describe()["decide"] == "jev:default"
    assert router(env={"AIMLAPI_KEY": "k"}).describe()["decide"] == "jev:default"
    assert router(config={"decide": "claude:haiku"}, env={"JEV_API_KEY": "k"}).describe()["decide"] == "claude:haiku"


def test_config_file_and_env_override(tmp_path):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"chat": "claude:opus", "options": {"claude_bin": "/x/claude"}}))
    r = ConfiguredModelRouter(config_path=str(p), env={"WORLDCAM_MODEL_CHAT": "gemini:gemini-2.5-pro",
                                                       "WORLDCAM_MODEL_SUMMARIZE": "claude:haiku"})
    d = r.describe()
    assert d["chat"] == "gemini:gemini-2.5-pro" and d["summarize"] == "claude:haiku"
    assert d["agent"] == "claude:sonnet" and r.options["claude_bin"] == "/x/claude"
    with pytest.raises(FileNotFoundError):
        ConfiguredModelRouter(config_path=str(tmp_path / "nope.json"), env={})


def test_bad_config_fails_early():
    with pytest.raises(ValueError, match="unknown task"):
        router({"vision": "gemini:x"})
    with pytest.raises(ValueError, match="backend:model"):
        router({"chat": "sonnet"})
    with pytest.raises(ValueError):
        parse_spec(":x")


def test_missing_credentials_raise_not_configured_without_fallback():
    r = router()
    with pytest.raises(NotConfigured, match="GEMINI_API_KEY"):
        r.client("vision_crop")
    with pytest.raises(NotConfigured, match="JEV_API_KEY"):
        router({"decide": "jev:default"}).client("decide")
    with pytest.raises(NotConfigured, match="ANTHROPIC_API_KEY"):
        router({"decide": "claude-api:claude-haiku-5-5"}).client("decide")
    with pytest.raises(NotConfigured, match="unknown backend"):
        router({"decide": "nope:x"}).client("decide")
    with pytest.raises(NotConfigured, match="claude"):
        router({"chat": "claude:sonnet", "options": {"claude_bin": "/definitely/not/here"}}).client("chat")


def test_capability_mismatch_is_explicit():
    r = router({"decide": "jev:default"}, env={"JEV_API_KEY": "k"})
    assert r.client("decide").name == "jev:typesafe/jev"
    with pytest.raises(NotConfigured, match="tool loops"):
        r.runner("decide")


def test_routes_and_records_usage_without_prompts(tmp_path):
    log = UsageLog(path=str(tmp_path / "u.jsonl"))
    r = router({"vision_crop": "gemini:gemini-2.5-flash", "chat": "claude:haiku"},
               env={"GEMINI_API_KEY": "k"}, usage=log,
               backend_kwargs={"gemini": {"transport": fake_gemini_transport}, "claude": {"spawn": fake_spawn}})
    out = r.client("vision_crop").generate(GenRequest("vision_crop", "SECRET PROMPT TEXT"))
    assert out.text == "a red box"
    assert r.client("chat").generate(GenRequest("chat", "hello")).text == "hi"
    s = log.summary()
    assert s["vision_crop"]["input_tokens"] == 1000 and s["vision_crop"]["cost_usd"] > 0
    assert s["chat"]["cost_usd"] == 0.01
    blob = (tmp_path / "u.jsonl").read_text()
    assert "SECRET" not in blob and len(blob.splitlines()) == 2
    assert r.client("chat") is r.client("chat")      # cached


def test_usage_log_bounded_and_failures_counted():
    log = UsageLog(maxlen=3)
    for i in range(10):
        log.record("t", "m", {"input_tokens": 1}, 5.0, ok=i % 2 == 0)
    assert len(log.records) == 3
    assert log.summary()["t"]["calls"] == 3
