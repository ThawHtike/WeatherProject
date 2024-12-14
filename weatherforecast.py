import requests

city="Yangon"

api_key = "a5bedda26f30751cafc720db15d72c20"

url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'

response =requests.get(url)

data = response.json()
print(data)

print('weather is ',data['weather'][0]['description'])

print('current temperature is ',data['main']['temp'])

print('current temperature feel like is ',data['main']['feels_like'])

print('current temperature humidity is ',data['main']['humidity'])
