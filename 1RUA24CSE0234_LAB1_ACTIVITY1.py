import requests
import csv
from datetime import datetime
import json

bengaluru_locations = [
    {"name": "City Center", "latitude": 12.9716, "longitude": 77.5946},
    {"name": "Whitefield", "latitude": 12.9698, "longitude": 77.7500},
    {"name": "Electronic City", "latitude": 12.8409, "longitude": 77.6749},
    {"name": "Jayanagar", "latitude": 12.9244, "longitude": 77.5830},
    {"name": "Yeshwanthpur", "latitude": 13.0242, "longitude": 77.5516},
]


def collect_bengaluru_weather_data():
    """
    Collects over 500 hourly weather data points for five locations in Bengaluru.
    This data includes temperature and relative humidity.
    """
    print("--- Collecting Real-Time Weather Data for Bengaluru from an API ---")

    api_url = "https://api.open-meteo.com/v1/forecast"

    all_weather_data = []

    for location in bengaluru_locations:
        print(f"Fetching weather data for {location['name']}...")


        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "hourly": ["temperature_2m", "relative_humidity_2m"],
            "timezone": "Asia/Kolkata",
            "past_days": 21
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            hourly_data = data['hourly']
            timestamps = hourly_data['time']
            temperatures = hourly_data['temperature_2m']
            humidities = hourly_data['relative_humidity_2m']

            for i in range(len(timestamps)):
                all_weather_data.append({
                    'location': location['name'],
                    'timestamp': timestamps[i],
                    'temperature_c': temperatures[i],
                    'relative_humidity_percent': humidities[i]
                })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {location['name']}: {e}")
            continue

    print(f"\nSuccessfully fetched a total of {len(all_weather_data)} data points for weather.")

    return all_weather_data


def collect_bengaluru_air_quality_data():
    """
    Collects over 500 hourly air quality data points for five locations in Bengaluru.
    This is more suitable for data science than scraping.
    """
    print("\n--- Collecting Real-Time Air Quality Data for Bengaluru from an API ---")


    api_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    all_air_quality_data = []

    for location in bengaluru_locations:
        print(f"Fetching air quality data for {location['name']}...")

      
        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "hourly": ["pm10", "pm2_5", "carbon_monoxide"],
            "timezone": "Asia/Kolkata",
            "past_days": 30 # This will give us plenty of data points
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            hourly_data = data['hourly']
            timestamps = hourly_data['time']
            pm10_values = hourly_data['pm10']
            pm2_5_values = hourly_data['pm2_5']
            co_values = hourly_data['carbon_monoxide']

            for i in range(len(timestamps)):
                all_air_quality_data.append({
                    'location': location['name'],
                    'timestamp': timestamps[i],
                    'pm10_ug_m3': pm10_values[i],
                    'pm2_5_ug_m3': pm2_5_values[i],
                    'carbon_monoxide_ug_m3': co_values[i]
                })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {location['name']}: {e}")
            continue

    print(f"\nSuccessfully fetched a total of {len(all_air_quality_data)} data points for air quality.")

    return all_air_quality_data



def save_to_csv(data, filename):
    """
    Saves a list of dictionaries to a CSV file.
    """
    if not data:
        print(f"No data to save to {filename}. Skipping.")
        return

    # Extract column headers from the keys of the first dictionary
    keys = data[0].keys()

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

        print(f"Data successfully saved to {filename}")

    except IOError as e:
        print(f"Error saving data to CSV file: {e}")


if __name__ == "__main__":
    print("--- Starting Bengaluru Data Collection and CSV Export ---")

    # 1. Collect weather data and save to CSV
    weather_data = collect_bengaluru_weather_data()
    if weather_data:
        save_to_csv(weather_data, 'bengaluru_weather.csv')

    # 2. Collect air quality data and save to CSV
    air_quality_data = collect_bengaluru_air_quality_data()
    if air_quality_data:
        save_to_csv(air_quality_data, 'bengaluru_air_quality.csv')

    print("\n--- Process Complete ---")