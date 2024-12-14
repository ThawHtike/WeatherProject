import requests

city = input('Enter your city name: ')

api_key = "a5bedda26f30751cafc720db15d72c20"

url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'

response = requests.get(url)

print(f"HTTP response status_code = {response.status_code}")

if response.status_code == 200:
    data = response.json()

    # Extract temperature
    temperature = data['main']['temp']
    temp_in_C = round(temperature - 273.15, 2)
    temp_in_F = round((temp_in_C * 9 / 5) + 32, 2)

    # Extract weather description
    description = data['weather'][0]['description']

    # Extract geographical location
    latitude = data['coord']['lat']
    longitude = data['coord']['lon']

    print(f"Current weather in {city}: {description}, Temperature: {temp_in_C} °C or {temp_in_F} °F")
    print(f"Geographical location: Latitude = {latitude}, Longitude = {longitude}")
else:
    print("City not found. Please check your input.")
