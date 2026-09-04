import httpx
import json

sample_text = """Human-Computer Interaction (HCI) has developed rapidly over the past few decades. Researchers and developers continue to look for natural, simple, and low-cost ways for people to interact with computers."""

url = "http://127.0.0.1:8000/humanize"

payload = {
    "text": sample_text
}

print(f"Sending request to {url}...")
try:
    response = httpx.post(url, json=payload, timeout=120.0)
    print(f"Status code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

