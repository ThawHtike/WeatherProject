import datetime as dt
import requests

city="Yangon"

api_key = "a5bedda26f30751cafc720db15d72c20"

url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'

response=requests.get(url).json()
print(response)