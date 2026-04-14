import requests

url = "http://127.0.0.1:5000/predict"

# Example single-sample payload (must include all 36 features)
payload = {
    "IAT": 0.02,
    "Min": 54.0,
    "Magnitue": 10.4,
    "fin_flag_number": 0.0,
    "psh_flag_number": 0.0,
    "syn_flag_number": 0.0,
    "Tot sum": 540.0,
    "Protocol Type": 6.0,
    "ICMP": 0.0,
    "Header_Length": 54.0,
    "rst_count": 0.0,
    "Radius": 0.75,
    "fin_count": 0.0,
    "syn_count": 0.0,
    "flow_duration": 0.05,
    "Srate": 10.5,
    "Number": 9.5,
    "AVG": 54.0,
    "Rate": 10.5,
    "Variance": 0.10,
    "HTTPS": 0.0,
    "urg_count": 0.0,
    "Duration": 64.0,
    "Weight": 141.55,
    "HTTP": 0.0,
    "Max": 54.0,
    "Tot size": 54.0,
    "Covariance": 0.5,
    "ack_count": 1.0,
    "Std": 0.25,
    "rst_flag_number": 0.0,
    "UDP": 0.0,
    "ack_flag_number": 1.0,
    "SSH": 0.0,
    "TCP": 1.0,
    "LLC": 1.0
}

response = requests.post(url, json=payload, timeout=10)
print("Status:", response.status_code)
print("Response:", response.json())
