from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from eclipse.panel.app import create_app
from eclipse.panel.controller import MockPipelineController


@pytest.fixture
def setup(tmp_path: Path):
    controller = MockPipelineController(config_path=tmp_path / "private" / "config.json")
    with TestClient(create_app(controller), base_url="http://127.0.0.1:8000") as client:
        client.headers["Authorization"] = "Bearer " + client.app.state.panel_token
        yield client, controller


def test_initial_mock_state(setup):
    client, _ = setup
    state = client.get("/api/state").json()
    assert state["simulated"] is True
    assert state["output_connected"] is False
    assert state["output_state"] == "stopped"
    assert len(state["enabled"]) == 13
    assert all(v == "unavailable" for v in state["detector_health"].values())


def test_panic_persists_until_resume(setup):
    client, _ = setup
    assert client.post("/api/panic").json()["panic"] is True
    assert client.get("/api/state").json()["panic"] is True
    assert client.post("/api/resume").json()["panic"] is False


def test_config_is_atomic_and_persists_keywords(setup):
    client, controller = setup
    good = {"enabled": ["two_fa", "email", "keyword"], "keywords": ["FAKE_DEMO_STREET"]}
    assert client.put("/api/config", json=good).status_code == 200
    assert controller._config_path.is_file()
    assert client.put("/api/config", json={"enabled": ["2fa"], "keywords": []}).status_code == 422
    assert client.get("/api/state").json()["enabled"] == good["enabled"]
    assert client.put("/api/config", json={"enabled": ["email"], "keywords": [""]}).status_code == 422
    assert client.get("/api/state").json()["keywords"] == good["keywords"]
    # Restart: only keywords persist, categories return to all enabled.
    restarted = MockPipelineController(controller._config_path)
    assert restarted.get_state().keywords == good["keywords"]
    assert len(restarted.get_state().enabled) == 13


def test_irl_rejected_and_desktop_accepted(setup):
    client, _ = setup
    assert client.put("/api/mode", json={"mode": "irl"}).status_code == 409
    assert client.put("/api/mode", json={"mode": "desktop"}).status_code == 200


def test_log_incremental_and_merged(setup):
    client, controller = setup
    controller.events.record("screen", "email", timestamp=100.0)
    controller.events.record("screen", "email", timestamp=100.2)
    body = client.get("/api/log?after=0").json()
    assert len(body["events"]) == 1
    assert body["events"][0]["count"] == 2
    last = body["last"]
    assert client.get(f"/api/log?after={last}").json()["events"] == []
    controller.events.record("webcam", "mail_label", timestamp=101.0)
    assert len(client.get(f"/api/log?after={last}").json()["events"]) == 1


def test_untrusted_origin_and_host_rejected(setup):
    client, _ = setup
    assert client.post("/api/panic", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert client.get("/api/state", headers={"Host": "attacker.example"}).status_code == 403
    assert client.get("/api/state", headers={"Origin": "http://127.0.0.1:9000"}).status_code == 403


def test_docs_and_health(setup):
    client, _ = setup
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/api/health").json()["ok"] is True


def test_frontend_served_with_assets(setup):
    client, _ = setup
    index = client.get("/")
    assert "Your privacy," in index.text
    assert "SIMULATION MODE" in index.text
    assert "/static/app.js" in index.text
    js = client.get("/static/app.js")
    css = client.get("/static/styles.css")
    assert js.status_code == 200 and "refreshState" in js.text
    assert css.status_code == 200 and ".hero" in css.text


def test_panel_does_not_contain_sample_sensitive_data(setup):
    client, _ = setup
    assert "FAKE_DEMO_STREET" not in client.get("/").text


def test_event_merging_survives_interleaved_categories(setup):
    client, controller = setup
    controller.events.record("screen", "email", timestamp=100.0)
    controller.events.record("screen", "phone", timestamp=100.1)
    controller.events.record("screen", "email", timestamp=100.4)
    events = client.get("/api/log?after=0").json()["events"]
    assert len(events) == 2
    assert next(e for e in events if e["category"] == "email")["count"] == 2
    assert next(e for e in events if e["category"] == "phone")["count"] == 1


def test_old_catches_create_separate_event_after_quiet_gap(setup):
    client, controller = setup
    controller.events.record("screen", "email", timestamp=100.0)
    controller.events.record("screen", "email", timestamp=105.0)
    events = client.get("/api/log?after=0").json()["events"]
    assert len(events) == 2
    assert [e["count"] for e in events] == [1, 1]


def test_event_history_is_bounded(setup):
    client, controller = setup
    for i in range(300):
        controller.events.record("screen", "email", timestamp=1000.0 + i * 4)
    events = client.get("/api/log?after=0").json()
    assert len(events["events"]) == 200
    assert events["last"] == 300


def test_activity_status_is_present_in_frontend(setup):
    client, _ = setup
    assert 'id="log-connection"' in client.get("/").text
    js = client.get("/static/app.js").text
    assert "Activity log unavailable" in js
    assert "Output unverified" in js


def test_incident_identity_survives_updates_but_not_quiet_gaps(setup):
    client, controller = setup
    first = controller.events.record('screen', 'email', timestamp=100)
    controller.events.record('screen', 'phone', timestamp=100.5)
    updated = controller.events.record('screen', 'email', timestamp=101)
    separate = controller.events.record('screen', 'email', timestamp=105)
    assert first.incident_id == updated.incident_id
    assert updated.seq > first.seq
    assert separate.incident_id != first.incident_id
    data = client.get('/api/log', params={'after': first.seq}).json()
    assert next(e for e in data['events'] if e['incident_id'] == first.incident_id)['count'] == 2


@pytest.mark.parametrize('new_count', [0, 1, 5])
def test_log_restart_returns_new_snapshot_regardless_of_old_cursor(setup, tmp_path, new_count):
    client, controller = setup
    for i in range(3):
        controller.events.record('screen', 'email', timestamp=i * 5)
    old = client.get('/api/log').json()
    restarted = MockPipelineController(config_path=tmp_path / 'restarted.json')
    for i in range(new_count):
        restarted.events.record('screen', 'phone', timestamp=i * 5)
    client.app.state.controller = restarted
    data = client.get('/api/log', params={'after': old['last'], 'session_id': old['session_id']}).json()
    assert data['session_id'] != old['session_id']
    assert len(data['events']) == new_count
    assert data['last'] == new_count
    assert client.get('/api/log', params={'after': data['last'], 'session_id': data['session_id']}).json()['events'] == []
    next_event = restarted.events.record('screen', 'card', timestamp=100)
    fresh = client.get('/api/log', params={'after': data['last'], 'session_id': data['session_id']}).json()
    assert [e['incident_id'] for e in fresh['events']] == [next_event.incident_id]


@pytest.mark.parametrize('authorization', [None, 'Bearer wrong-token'])
def test_private_routes_require_launch_authorization(setup, authorization):
    client, controller = setup
    before = controller.get_state()
    client.headers.pop('Authorization')
    headers = {} if authorization is None else {'Authorization': authorization}
    requests = [
        ('GET', '/api/state', None), ('GET', '/api/log', None),
        ('PUT', '/api/config', {'enabled': [], 'keywords': ['FAKE_PRIVATE']}),
        ('POST', '/api/panic', None), ('POST', '/api/resume', None),
        ('PUT', '/api/mode', {'mode': 'desktop'}),
    ]
    for method, path, body in requests:
        response = client.request(method, path, json=body, headers=headers)
        assert response.status_code == 401
        assert response.headers['cache-control'] == 'no-store'
    assert controller.get_state() == before
    assert not controller._config_path.exists()
    assert client.get('/').status_code == 200
    assert client.get('/api/health').status_code == 200
    assert client.app.state.panel_token not in client.get('/').text
    assert client.app.state.panel_token not in client.get('/static/app.js').text


def test_old_launch_token_cannot_resume_new_session(tmp_path):
    first = create_app(MockPipelineController(tmp_path / 'first.json'))
    controller = MockPipelineController(tmp_path / 'second.json')
    second = create_app(controller)
    controller.panic()
    with TestClient(second, base_url='http://127.0.0.1:8000') as client:
        response = client.post('/api/resume', headers={'Authorization': 'Bearer ' + first.state.panel_token})
        assert response.status_code == 401
        assert controller.get_state().panic is True
        response = client.post('/api/resume', headers={'Authorization': 'Bearer ' + second.state.panel_token})
        assert response.status_code == 200
        assert response.json()['panic'] is False
        assert response.headers['referrer-policy'] == 'no-referrer'
