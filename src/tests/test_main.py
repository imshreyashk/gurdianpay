from fastapi.testclient import TestClient
import sys
import os

# This helps the test find main.py in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Online" in response.json()["message"]