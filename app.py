
import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Modèle introuvable. Lance d'abord : python train.py")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle["model"]
le_weather = bundle["le_weather"]
le_city = bundle["le_city"]
features = bundle["features"]
metrics = bundle["metrics"]
cities = bundle["cities"]
city_descriptions = bundle.get("city_descriptions", {})

WEATHER_LABELS = {
    "clear": "☀️ Dégagé",
    "rain": "🌧️ Pluie",
    "fog": "🌫️ Brouillard",
    "snow": "❄️ Neige"
}

DAY_LABELS = [
    "Lundi", "Mardi", "Mercredi", "Jeudi",
    "Vendredi", "Samedi", "Dimanche"
]

# Noms des villes en français avec drapeau
CITY_LABELS = {
    "casablanca": "🇲🇦 Casablanca",
    "rabat": "🇲🇦 Rabat",
    "tanger": "🇲🇦 Tanger",
    "fes": "🇲🇦 Fès",
    "marrakesh": "🇲🇦 Marrakech",
    "agadir": "🇲🇦 Agadir",
    "meknes": "🇲🇦 Meknès",
    "oujda": "🇲🇦 Oujda",
    "kenitra": "🇲🇦 Kénitra",
    "tetouan": "🇲🇦 Tétouan",
    "safi": "🇲🇦 Safi",
    "el_jadida": "🇲🇦 El Jadida",
    "nador": "🇲🇦 Nador",
    "settat": "🇲🇦 Settat",
    "berrechid": "🇲🇦 Berrechid",
    "khouribga": "🇲🇦 Khouribga",
    "beni_mellal": "🇲🇦 Béni Mellal",
    "essaouira": "🇲🇦 Essaouira",
    "larache": "🇲🇦 Larache",
    "al_hoceima": "🇲🇦 Al Hoceïma",
    "laayoune": "🇲🇦 Laâyoune",
    "dakhla": "🇲🇦 Dakhla",
    "guelmim": "🇲🇦 Guelmim",
    "tarfaya": "🇲🇦 Tarfaya",
    "chefchaouen": "🇲🇦 Chefchaouen",
    "ouzoud": "🇲🇦 Ouzoud",
    "ifrane": "🇲🇦 Ifrane"
}

def classify_volume(v):
    if v < 300:
        return "Faible", "low"
    if v < 700:
        return "Modéré", "medium"
    if v < 1000:
        return "Élevé", "high"
    return "Très élevé", "critical"

def build_features(city, hour, day, weather, is_holiday, temperature):
    city_enc = le_city.transform([city])[0]
    weather_enc = le_weather.transform([weather])[0]
    is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_weekend = 1 if day >= 5 else 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    return np.array([[
        city_enc, hour, day, weather_enc,
        is_holiday, temperature,
        is_rush_hour, is_weekend,
        hour_sin, hour_cos
    ]])

@app.route("/")
def index():
    return render_template(
        "index.html",
        weather_options=WEATHER_LABELS,
        day_labels=DAY_LABELS,
        city_labels=CITY_LABELS,
        cities=cities,
        city_descriptions=city_descriptions,
        metrics=metrics
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form["city"]
        hour = int(request.form["hour"])
        day = int(request.form["day_of_week"])
        weather = request.form["weather"]
        is_holiday = int(request.form.get("is_holiday", 0))
        temperature = float(request.form.get("temperature", 25))  # Température par défaut Maroc

        if not (0 <= hour <= 23):
            raise ValueError("L'heure doit être entre 0 et 23.")
        if city not in le_city.classes_:
            raise ValueError(f"Ville invalide : {city}")
        if weather not in le_weather.classes_:
            raise ValueError(f"Météo invalide : {weather}")

        X = build_features(city, hour, day, weather, is_holiday, temperature)
        volume = round(model.predict(X)[0])
        label, level = classify_volume(volume)

        return jsonify({
            "success": True,
            "volume": volume,
            "label": label,
            "level": level,
            "city": CITY_LABELS.get(city, city),
            "hour": hour,
            "day": DAY_LABELS[day],
            "weather": WEATHER_LABELS[weather],
            "is_holiday": bool(is_holiday),
            "temperature": temperature
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/hourly", methods=["POST"])
def hourly_forecast():
    try:
        data = request.json
        city = data.get("city", "casablanca")
        day = int(data.get("day_of_week", 0))
        weather = data.get("weather", "clear")
        is_holiday = int(data.get("is_holiday", 0))
        temperature = float(data.get("temperature", 25))

        forecast = []
        for h in range(24):
            X = build_features(city, h, day, weather, is_holiday, temperature)
            v = round(model.predict(X)[0])
            _, level = classify_volume(v)
            forecast.append({"hour": h, "volume": v, "level": level})

        return jsonify({"success": True, "forecast": forecast})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    print("=" * 50)
    print("🚦 TrafficAI - MAROC 🇲🇦")
    print("=" * 50)
    print(f"   Villes : {', '.join(cities)}")
    print(f"   MAE : {metrics['mae']:.1f} véhicules | R² : {metrics['r2']:.3f}")
    print("   → http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)