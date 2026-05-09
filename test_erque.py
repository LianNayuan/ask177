import requests, json

r = requests.post("http://localhost:8003/ask", json={
    "question": "二确武器有哪些",
    "session_id": 0
}, timeout=120)
data = r.json()
# Check if 审查者 is in the answer
has_shencha = "审查者" in data['answer']
print(f"Contains 审查者: {has_shencha}")
print()
print(data['answer'])
