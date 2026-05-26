import json
from app import create_app
app = create_app('production')
with app.test_client() as client:
    resp = client.get('/api/health')
    print(json.dumps(resp.get_json()))
