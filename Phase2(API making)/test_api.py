import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "flow_duration": 0.05,
    "Header_Length": 54,
    "Protocol Type": 6,
    "Rate": 10.5,
    "Srate": 10.5,
    "ack_count": 1,
    "syn_count": 0,
    "rst_count": 0,
    "Tot size": 54,
    "IAT": 0.02
}

response = requests.post(url, json=payload)
print(response.json())