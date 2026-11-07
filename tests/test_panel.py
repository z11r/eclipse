from fastapi.testclient import TestClient
from eclipse.panel.app import create_app


def test_health():
    with TestClient(create_app()) as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json() == {'service': 'eclipse-panel', 'ok': True, 'pipeline_mode': 'mock'}


def test_status_is_explicitly_mock_and_stopped():
    with TestClient(create_app()) as client:
        response = client.get('/api/status')
        assert response.status_code == 200
        state = response.json()
        assert state['state'] == 'stopped'
        assert state['output_connected'] is False
        assert state['mode'] == 'mock'
        assert all(detector['state'] == 'unavailable' for detector in state['detectors'].values())
