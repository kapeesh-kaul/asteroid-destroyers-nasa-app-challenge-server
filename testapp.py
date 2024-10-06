import requests
from numpy import inf
url = 'http://127.0.0.1:5001/get_top_planets'
data = {
    # "filepath": 'PSCompPars.csv',
    # "SNR0": 100,
    # "D": 6,
    "top_n": 2,
    "SNR_filter": -10000000000000,
    # "category": "hello"
}

response = requests.post(url, json=data)
print(response.json())
