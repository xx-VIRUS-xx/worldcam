import base64
import json

import numpy as np
import pytest

from worldcam.ambiguity.engine import AmbiguityEngine, EngineConfig
from worldcam.ambiguity.entities import EntityObserver, classify_label, make_entity_key
from worldcam.ambiguity.file_store import FileGraphStore
from worldcam.ambiguity.gemini import GeminiClient, GeminiError, GeminiWorkerRunner, redact_key
from worldcam.ambiguity.triggers import DoubtDetector, IdentitySignal, TriggerConfig
from worldcam.ambiguity.worker import ClaudeWorkerRunner, make_worker_runner
from worldcam.graph.api import Doubt, InMemoryGraphStore, Sighting, new_id
from worldcam.messenger.base import FakeNotifier, OwnerReply
from worldcam.pipeline import BestSnapshot, Track, TrackState


class FakeGeminiTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": json.loads(body.decode("utf-8")),
                           "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, payload = item
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        return status, raw


def _cand(parts):
    return {"candidates": [{"content": {"role": "model", "parts": parts}}],
            "usageMetadata": {"totalTokenCount": 42}}


def _mk_snaps(tmp_path):
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    crop = snaps / "dog_001.jpg"
    crop.write_bytes(b"\xff\xd8\xff\xe0fakejpg\xff\xd9")
    return snaps, crop


def test_gemini_client_redacts_key_and_retries():
    tr = FakeGeminiTransport([
        (429, {"error": {"message": "rate limited key=AIzaSyDummyKey1234567890123456"}}),
        (200, _cand([{"text": "{\"action\":\"asked\",\"summary\":\"ok\"}"}])),
    ])
    sleeps = []
    client = GeminiClient("AIzaSyDummyKey1234567890123456", transport=tr, sleep=sleeps.append)
    assert "AIza" not in repr(client)
    res = client.generate_content({"contents": []})
    assert "candidates" in res and len(sleeps) == 1
    assert tr.calls[0]["headers"]["x-goog-api-key"] == "AIzaSyDummyKey1234567890123456"
    assert "key=" not in tr.calls[0]["url"]

    tr_bad = FakeGeminiTransport([(400, {"error": {"message": "bad key AIzaSyDummyKey1234567890123456"}})])
    c2 = GeminiClient("AIzaSyDummyKey1234567890123456", transport=tr_bad, sleep=lambda _: None)
    with pytest.raises(GeminiError) as ei:
        c2.generate_content({"contents": []})
    assert "AIzaSy" not in str(ei.value)
    assert redact_key("err AIzaSyDummyKey1234567890123456") == "err <redacted>"


def test_gemini_worker_ask_and_resolve_animal_persona(tmp_path):
    snaps, crop = _mk_snaps(tmp_path)
    store = InMemoryGraphStore()
    owner_id = store.upsert_person(name="Owner", kind="owner")
    sid = new_id("s")
    now = 1795500000.0
    store.add_sighting(Sighting(sid, now, "door", 5, zones=("door",), snapshot_path=str(crop),
                                entity_label="dog"))
    did = DoubtDetector(store, TriggerConfig(entity_min_sightings=1)).observe(IdentitySignal(
        kind="unknown_entity", track_id=5, crop_path=str(crop), ts=now,
        entity_type="animal", entity_label="dog", entity_key="ek_animal_dog_door",
        sighting_id=sid, zones=("door",),
    ), now=now)
    assert did is not None

    # Mock Gemini multi-turn tool calling for ask -> resolve, including thoughtSignature preservation
    tr = FakeGeminiTransport([
        # Ask run: turn 1 get_doubt + search_people; turn 2 ask_owner; turn 3 final JSON
        (200, _cand([
            {"functionCall": {"name": "get_doubt", "args": {"doubt_id": did}}, "thoughtSignature": "sig1"},
            {"functionCall": {"name": "search_people", "args": {"text": "dog"}}},
        ])),
        (200, _cand([
            {"functionCall": {"name": "ask_owner", "args": {
                "doubt_id": did,
                "text": "I spotted a dog near the door. Is this your pet, and what is their name?",
                "choices": ["New entity", "Ignore"],
            }}},
        ])),
        (200, _cand([{"text": json.dumps({"action": "asked", "summary": "Asked about the dog at the door"})}])),
        # Resolve run: turn 1 get_doubt; turn 2 upsert_person + add_fact + relate + set_persona + resolve;
        # turn 3 prose -> triggers structured-output finalization turn 4
        (200, _cand([{"functionCall": {"name": "get_doubt", "args": {"doubt_id": did}}}])),
        (200, _cand([
            {"functionCall": {"name": "upsert_person", "args": {
                "name": "Bruno", "kind": "pet", "entity_type": "animal", "label": "dog",
            }}},
        ])),
        (200, _cand([
            {"functionCall": {"name": "search_people", "args": {"text": "Owner"}}},
        ])),
        (200, _cand([
            {"functionCall": {"name": "add_fact", "args": {
                "person_id": "__BRUNO__", "key": "species_or_breed", "value": "golden retriever",
                "source": "owner", "confidence": 0.9,
            }}},
            {"functionCall": {"name": "add_fact", "args": {
                "person_id": "__BRUNO__", "key": "color", "value": "golden coat",
                "source": "observed", "confidence": 0.8,
            }}},
            {"functionCall": {"name": "relate", "args": {
                "a_id": "__BRUNO__", "relation": "pet_of", "b_id": owner_id, "source": "owner",
            }}},
            {"functionCall": {"name": "set_persona_summary", "args": {
                "person_id": "__BRUNO__",
                "summary": "Bruno is the owner's golden retriever dog with a golden coat, often seen near the door.",
            }}},
            {"functionCall": {"name": "resolve_doubt", "args": {
                "doubt_id": did, "summary": "Recorded pet Bruno (golden retriever).",
            }}},
        ])),
        (200, _cand([{"text": "All done! Updated Bruno."}])),
        (200, _cand([{"text": json.dumps({
            "action": "resolved", "summary": "Created animal Bruno and built persona.",
            "person_ids": ["p_bruno"], "facts_written": 2,
        })}])),
    ])

    # Patch the placeholder __BRUNO__ dynamically once upsert_person creates the real id
    orig_post = tr.post

    def dynamic_post(url, headers, body, timeout):
        payload = json.loads(body.decode("utf-8"))
        # Look for created person_id in previous tool responses
        bruno_id = None
        for msg in payload.get("contents", []):
            for p in msg.get("parts", []):
                fr = p.get("functionResponse")
                if fr and fr.get("name") == "upsert_person":
                    bruno_id = (fr.get("response") or {}).get("result", {}).get("person_id")
        status, raw = orig_post(url, headers, body, timeout)
        if bruno_id:
            raw = raw.replace(b"__BRUNO__", bruno_id.encode("utf-8"))
        return status, raw

    tr.post = dynamic_post

    runner = GeminiWorkerRunner(
        store=store, snapshot_dir=str(snaps), api_key="test-key", model="gemini-2.5-flash", transport=tr,
    )
    notifier = FakeNotifier()
    eng = AmbiguityEngine(store, notifier, runner, EngineConfig(snapshot_dir=str(snaps), quiet_start_hour=0,
                                                                quiet_end_hour=0))
    rep1 = eng.tick(now + 1)
    assert rep1.asked_runs == [did] and rep1.sent == [did]

    # Verify multimodal image was attached & thoughtSignature preserved in ask loop
    first_call_parts = tr.calls[0]["payload"]["contents"][0]["parts"]
    assert any(p.get("inlineData", {}).get("mimeType") == "image/jpeg"
               and base64.b64decode(p["inlineData"]["data"]).startswith(b"\xff\xd8")
               for p in first_call_parts)
    assert tr.calls[1]["payload"]["contents"][1]["parts"][0]["thoughtSignature"] == "sig1"

    ref, q = notifier.sent[0]
    assert eng.handle_reply(OwnerReply(ref, text="That's Bruno, our golden retriever!"), now + 10) == did
    rep2 = eng.tick(now + 11)
    assert rep2.resolve_runs == [did]

    # Verify structured-output finalizer turn dropped `tools` and passed `responseJsonSchema`
    fin_payload = tr.calls[-1]["payload"]
    assert "tools" not in fin_payload
    assert fin_payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseJsonSchema" in fin_payload["generationConfig"]

    d = store.get_doubt(did)
    assert d.status == "resolved"
    bruno = store.search_people("Bruno")[0]
    assert bruno.entity_type == "animal" and bruno.label == "dog" and bruno.kind == "pet"
    assert "golden retriever" in bruno.persona_summary
    assert [(f.key, f.source) for f in store.facts(bruno.id)] == [
        ("species_or_breed", "owner"), ("color", "observed"),
    ]
    assert store.relations(bruno.id) == [("pet_of", owner_id, "out")]
    assert store.recent_sightings(person_id=bruno.id)[0].id == sid


def test_gemini_worker_ignores_images_outside_snapshot_dir_and_recovers_from_tool_error(tmp_path):
    snaps, crop = _mk_snaps(tmp_path)
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"secret")
    store = FileGraphStore(str(tmp_path / "g.pkl"))
    did = store.create_doubt(Doubt(
        id="d_obj", kind="unknown_entity", about_ids=("s_1",),
        snapshot_paths=(str(outside), str(crop)),
        detail={"entity_type": "object", "entity_label": "backpack"},
    ))
    store.add_sighting(Sighting("s_1", 10.0, "door", 3, snapshot_path=str(crop), entity_label="backpack"))

    tr = FakeGeminiTransport([
        # Bad tool call first (wrong doubt id), then valid ask_owner, then JSON response
        (200, _cand([{"functionCall": {"name": "get_doubt", "args": {"doubt_id": "d_other"}}}])),
        (200, _cand([{"functionCall": {"name": "ask_owner", "args": {
            "doubt_id": did, "text": "Whose backpack is by the door?", "choices": ["Ignore"],
        }}}])),
        (200, _cand([{"text": json.dumps({"action": "asked", "summary": "Asked about backpack"})}])),
    ])
    runner = GeminiWorkerRunner(
        f"file:{tmp_path / 'g.pkl'}", str(snaps), api_key="k", transport=tr,
    )
    res = runner.run("ask", store.get_doubt(did))
    assert res.ok and res.action == "asked"
    user_parts = tr.calls[0]["payload"]["contents"][0]["parts"]
    # Only the confined crop inside snaps was attached (not `outside`)
    imgs = [p for p in user_parts if "inlineData" in p]
    assert len(imgs) == 1
    assert "only act on doubt" in tr.calls[1]["payload"]["contents"][2]["parts"][0]["functionResponse"]["response"]["error"]


def test_entity_observer_records_animals_and_objects_with_cooldown(tmp_path):
    snaps = tmp_path / "snaps"
    store = InMemoryGraphStore()
    det = DoubtDetector(store, TriggerConfig(entity_min_sightings=1))
    obs = EntityObserver(store, det, snapshot_dir=str(snaps), cooldown_s=60.0)

    assert classify_label("dog") == "animal"
    assert classify_label("backpack") == "object"
    assert classify_label("person") == "person"
    assert make_entity_key("object", "cell phone", ("desk",)) == "ek_object_cell_phone_desk"

    crop = np.zeros((48, 48, 3), np.uint8)
    t_dog = Track(1, "dog", TrackState.ACTIVE, (0.1, 0.2, 0.4, 0.5), 100.0, 101.0, zones=("door",),
                  best=BestSnapshot(seq=10, ts=101.0, box=(0.1, 0.2, 0.4, 0.5), score=0.9, crop=crop))
    t_person = Track(2, "person", TrackState.ACTIVE, (0.1, 0.2, 0.4, 0.5), 100.0, 101.0,
                     best=BestSnapshot(seq=11, ts=101.0, box=(0.1, 0.2, 0.4, 0.5), score=0.9, crop=crop))
    t_couch = Track(3, "couch", TrackState.STATIONARY, (0.1, 0.2, 0.4, 0.5), 100.0, 101.0,
                    best=BestSnapshot(seq=12, ts=101.0, box=(0.1, 0.2, 0.4, 0.5), score=0.9, crop=crop))

    assert obs.observe(t_person) is None
    assert obs.observe(t_couch) is None
    did = obs.observe(t_dog)
    assert did is not None
    d = store.get_doubt(did)
    assert d.kind == "unknown_entity" and d.detail["entity_type"] == "animal" and d.detail["entity_label"] == "dog"
    assert d.snapshot_paths and d.snapshot_paths[0].startswith(str(snaps))

    # Same track / seq is skipped; another dog track within cooldown is also skipped
    assert obs.observe(t_dog) is None
    t_dog2 = Track(4, "dog", TrackState.ACTIVE, (0.1, 0.2, 0.4, 0.5), 120.0, 120.0, zones=("door",),
                   best=BestSnapshot(seq=20, ts=120.0, box=(0.1, 0.2, 0.4, 0.5), score=0.9, crop=crop))
    assert obs.observe(t_dog2) is None
    assert obs.stats["cooldown_skips"] == 1


def test_make_worker_runner_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("worldcam.ambiguity.gemini.load_env", lambda *a, **kw: {})
    assert isinstance(make_worker_runner("auto", f"file:{tmp_path}/g.pkl", str(tmp_path)), ClaudeWorkerRunner)
    assert isinstance(make_worker_runner("gemini", f"file:{tmp_path}/g.pkl", str(tmp_path)), GeminiWorkerRunner)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(make_worker_runner("auto", f"file:{tmp_path}/g.pkl", str(tmp_path)), GeminiWorkerRunner)
    assert isinstance(make_worker_runner("claude", f"file:{tmp_path}/g.pkl", str(tmp_path)), ClaudeWorkerRunner)
