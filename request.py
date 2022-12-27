import requests



url = 'http://localhost:5000/api'
r = requests.post(url, json={'sentence':'행복해요정말.'})
print(r.json())