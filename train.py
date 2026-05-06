
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("model", exist_ok=True)

CITY_PROFILES = {
    
    "casablanca": {
        "rush_hour_factor": 1.6,
        "night_factor": 0.15,
        "weekend_factor": 0.55,
        "holiday_factor": 0.35,
        "rain_factor": 0.85,
        "snow_factor": 0.70,
        "base_volume": 800,
        "description": "🇲🇦 Casablanca - Capitale économique, plus grande ville"
    },
    "rabat": {
        "rush_hour_factor": 1.4,
        "night_factor": 0.12,
        "weekend_factor": 0.60,
        "holiday_factor": 0.40,
        "rain_factor": 0.88,
        "snow_factor": 0.75,
        "base_volume": 700,
        "description": "🇲🇦 Rabat - Capitale administrative"
    },
    "tanger": {
        "rush_hour_factor": 1.45,
        "night_factor": 0.13,
        "weekend_factor": 0.58,
        "holiday_factor": 0.38,
        "rain_factor": 0.86,
        "snow_factor": 0.72,
        "base_volume": 720,
        "description": "🇲🇦 Tanger - Détroit de Gibraltar, port stratégique"
    },
    "fes": {
        "rush_hour_factor": 1.35,
        "night_factor": 0.14,
        "weekend_factor": 0.62,
        "holiday_factor": 0.42,
        "rain_factor": 0.87,
        "snow_factor": 0.73,
        "base_volume": 650,
        "description": "🇲🇦 Fès - Capitale spirituelle et culturelle"
    },
    "marrakesh": {
        "rush_hour_factor": 1.38,
        "night_factor": 0.18,
        "weekend_factor": 0.61,
        "holiday_factor": 0.43,
        "rain_factor": 0.89,
        "snow_factor": 0.74,
        "base_volume": 680,
        "description": "🇲🇦 Marrakech - Capitale touristique"
    },
    "agadir": {
        "rush_hour_factor": 1.25,
        "night_factor": 0.20,
        "weekend_factor": 0.65,
        "holiday_factor": 0.45,
        "rain_factor": 0.92,
        "snow_factor": 0.80,
        "base_volume": 550,
        "description": "🇲🇦 Agadir - Station balnéaire"
    },
    "meknes": {
        "rush_hour_factor": 1.30,
        "night_factor": 0.14,
        "weekend_factor": 0.63,
        "holiday_factor": 0.42,
        "rain_factor": 0.88,
        "snow_factor": 0.75,
        "base_volume": 580,
        "description": "🇲🇦 Meknès - Ville impériale"
    },
    "oujda": {
        "rush_hour_factor": 1.28,
        "night_factor": 0.14,
        "weekend_factor": 0.64,
        "holiday_factor": 0.43,
        "rain_factor": 0.89,
        "snow_factor": 0.76,
        "base_volume": 560,
        "description": "🇲🇦 Oujda - Porte de l'Oriental"
    },
    "kenitra": {
        "rush_hour_factor": 1.32,
        "night_factor": 0.13,
        "weekend_factor": 0.62,
        "holiday_factor": 0.41,
        "rain_factor": 0.87,
        "snow_factor": 0.74,
        "base_volume": 600,
        "description": "🇲🇦 Kénitra - Ville industrielle"
    },
    "tetouan": {
        "rush_hour_factor": 1.35,
        "night_factor": 0.14,
        "weekend_factor": 0.61,
        "holiday_factor": 0.42,
        "rain_factor": 0.88,
        "snow_factor": 0.75,
        "base_volume": 590,
        "description": "🇲🇦 Tétouan - Perle du Nord"
    },
    "safi": {
        "rush_hour_factor": 1.20,
        "night_factor": 0.16,
        "weekend_factor": 0.66,
        "holiday_factor": 0.46,
        "rain_factor": 0.91,
        "snow_factor": 0.82,
        "base_volume": 500,
        "description": "🇲🇦 Safi - Capitale de la poterie"
    },
    "el_jadida": {
        "rush_hour_factor": 1.22,
        "night_factor": 0.15,
        "weekend_factor": 0.65,
        "holiday_factor": 0.45,
        "rain_factor": 0.90,
        "snow_factor": 0.81,
        "base_volume": 520,
        "description": "🇲🇦 El Jadida - Ville portuaire historique"
    },
    "nador": {
        "rush_hour_factor": 1.26,
        "night_factor": 0.15,
        "weekend_factor": 0.64,
        "holiday_factor": 0.44,
        "rain_factor": 0.89,
        "snow_factor": 0.78,
        "base_volume": 540,
        "description": "🇲🇦 Nador - Port de la Méditerranée"
    },
    "settat": {
        "rush_hour_factor": 1.24,
        "night_factor": 0.14,
        "weekend_factor": 0.65,
        "holiday_factor": 0.44,
        "rain_factor": 0.90,
        "snow_factor": 0.80,
        "base_volume": 530,
        "description": "🇲🇦 Settat - Ville agricole"
    },
    "berrechid": {
        "rush_hour_factor": 1.23,
        "night_factor": 0.14,
        "weekend_factor": 0.65,
        "holiday_factor": 0.44,
        "rain_factor": 0.90,
        "snow_factor": 0.80,
        "base_volume": 520,
        "description": "🇲🇦 Berrechid - Ville industrielle"
    },
    "khouribga": {
        "rush_hour_factor": 1.21,
        "night_factor": 0.15,
        "weekend_factor": 0.66,
        "holiday_factor": 0.45,
        "rain_factor": 0.91,
        "snow_factor": 0.82,
        "base_volume": 500,
        "description": "🇲🇦 Khouribga - Bassin minier"
    },
    "beni_mellal": {
        "rush_hour_factor": 1.22,
        "night_factor": 0.15,
        "weekend_factor": 0.65,
        "holiday_factor": 0.45,
        "rain_factor": 0.90,
        "snow_factor": 0.81,
        "base_volume": 510,
        "description": "🇲🇦 Béni Mellal - Ville du Tadla"
    },
    "essaouira": {
        "rush_hour_factor": 1.18,
        "night_factor": 0.22,
        "weekend_factor": 0.68,
        "holiday_factor": 0.48,
        "rain_factor": 0.93,
        "snow_factor": 0.85,
        "base_volume": 480,
        "description": "🇲🇦 Essaouira - Ville des alizés"
    },
    "larache": {
        "rush_hour_factor": 1.20,
        "night_factor": 0.16,
        "weekend_factor": 0.66,
        "holiday_factor": 0.46,
        "rain_factor": 0.91,
        "snow_factor": 0.82,
        "base_volume": 490,
        "description": "🇲🇦 Larache - Ville méditerranéenne"
    },
    "al_hoceima": {
        "rush_hour_factor": 1.17,
        "night_factor": 0.18,
        "weekend_factor": 0.67,
        "holiday_factor": 0.47,
        "rain_factor": 0.92,
        "snow_factor": 0.84,
        "base_volume": 470,
        "description": "🇲🇦 Al Hoceïma - Perle de la Méditerranée"
    },
    "laayoune": {
        "rush_hour_factor": 1.15,
        "night_factor": 0.17,
        "weekend_factor": 0.68,
        "holiday_factor": 0.48,
        "rain_factor": 0.94,
        "snow_factor": 0.86,
        "base_volume": 450,
        "description": "🇲🇦 Laâyoune - Ville du Sahara"
    },
    "dakhla": {
        "rush_hour_factor": 1.12,
        "night_factor": 0.20,
        "weekend_factor": 0.70,
        "holiday_factor": 0.50,
        "rain_factor": 0.95,
        "snow_factor": 0.88,
        "base_volume": 420,
        "description": "🇲🇦 Dakhla - Capitale du kitesurf"
    },
    "guelmim": {
        "rush_hour_factor": 1.14,
        "night_factor": 0.16,
        "weekend_factor": 0.69,
        "holiday_factor": 0.49,
        "rain_factor": 0.94,
        "snow_factor": 0.86,
        "base_volume": 440,
        "description": "🇲🇦 Guelmim - Porte du désert"
    },
    "tarfaya": {
        "rush_hour_factor": 1.10,
        "night_factor": 0.18,
        "weekend_factor": 0.71,
        "holiday_factor": 0.51,
        "rain_factor": 0.96,
        "snow_factor": 0.89,
        "base_volume": 400,
        "description": "🇲🇦 Tarfaya - Ville d'Antoine de Saint-Exupéry"
    },
    "chefchaouen": {
        "rush_hour_factor": 1.16,
        "night_factor": 0.21,
        "weekend_factor": 0.69,
        "holiday_factor": 0.49,
        "rain_factor": 0.93,
        "snow_factor": 0.85,
        "base_volume": 460,
        "description": "🇲🇦 Chefchaouen - La ville bleue"
    },
    "ouzoud": {
        "rush_hour_factor": 1.13,
        "night_factor": 0.19,
        "weekend_factor": 0.70,
        "holiday_factor": 0.50,
        "rain_factor": 0.94,
        "snow_factor": 0.87,
        "base_volume": 430,
        "description": "🇲🇦 Ouzoud - Célèbre pour ses cascades"
    },
    "ifrane": {
        "rush_hour_factor": 1.15,
        "night_factor": 0.18,
        "weekend_factor": 0.69,
        "holiday_factor": 0.49,
        "rain_factor": 0.93,
        "snow_factor": 0.85,
        "base_volume": 450,
        "description": "🇲🇦 Ifrane - Suisse marocaine"
    }
}

print("=" * 60)
print("🇲🇦 GÉNÉRATION DES DONNÉES POUR LES VILLES MAROCAINES")
print("=" * 60)

# Générer des données pour toutes les villes
np.random.seed(42)
n_samples_per_city = 2000

all_data = []

for city, profile in CITY_PROFILES.items():
    print(f"📊 {city.upper()} : {profile['description']}")
    
    for _ in range(n_samples_per_city):
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        weather = np.random.choice(['clear', 'rain', 'fog', 'snow'], p=[0.7, 0.15, 0.1, 0.05])
        is_holiday = np.random.choice([0, 1], p=[0.95, 0.05])
        temperature = np.random.uniform(5, 45)
        
        # Calcul du volume de base selon l'heure
        if 7 <= hour <= 9:
            base_volume = profile["base_volume"] * profile["rush_hour_factor"] + np.random.normal(0, 80)
        elif 17 <= hour <= 19:
            base_volume = profile["base_volume"] * profile["rush_hour_factor"] + np.random.normal(0, 80)
        elif 12 <= hour <= 14:
            base_volume = profile["base_volume"] * 0.7 + np.random.normal(0, 60)
        elif hour < 6 or hour > 22:
            base_volume = profile["base_volume"] * profile["night_factor"] + np.random.normal(0, 30)
        else:
            base_volume = profile["base_volume"] * 0.6 + np.random.normal(0, 50)
        
        if day_of_week >= 5:
            base_volume *= profile["weekend_factor"]
        
        weather_impact = {
            'clear': 1.0,
            'fog': 0.85,
            'rain': profile["rain_factor"],
            'snow': profile["snow_factor"]
        }
        base_volume *= weather_impact[weather]
        
        temp_impact = 1 - abs(temperature - 25) / 40
        temp_impact = max(0.5, min(1.2, temp_impact))
        base_volume *= temp_impact
        
        if is_holiday:
            base_volume *= profile["holiday_factor"]
        
        base_volume += np.random.normal(0, 25)
        volume = max(0, int(base_volume))
        
        all_data.append([city, hour, day_of_week, weather, is_holiday, temperature, volume])

# Créer DataFrame
df = pd.DataFrame(all_data, columns=['city', 'hour', 'day_of_week', 'weather', 'is_holiday', 'temperature', 'traffic_volume'])

print("\n" + "=" * 60)
print(f"✅ {len(df)} lignes générées pour {len(CITY_PROFILES)} villes marocaines")
print("=" * 60)

# Encodage
le_weather = LabelEncoder()
le_city = LabelEncoder()

df["weather_enc"] = le_weather.fit_transform(df["weather"])
df["city_enc"] = le_city.fit_transform(df["city"])

# Features
df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0)
df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

FEATURES = ["city_enc", "hour", "day_of_week", "weather_enc", "is_holiday", 
            "temperature", "is_rush_hour", "is_weekend", "hour_sin", "hour_cos"]

X = df[FEATURES]
y = df["traffic_volume"]

# Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 RÉSULTATS DU MODÈLE MAROCAIN :")
print(f"   MAE : {mae:.1f} véhicules")
print(f"   R²  : {r2:.3f}")

# Sauvegarde
bundle = {
    "model": model,
    "le_weather": le_weather,
    "le_city": le_city,
    "features": FEATURES,
    "cities": list(CITY_PROFILES.keys()),
    "city_descriptions": {city: profile["description"] for city, profile in CITY_PROFILES.items()},
    "metrics": {"mae": mae, "r2": r2}
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n" + "=" * 60)
print(f"✅ Modèle sauvegardé dans model/model.pkl")
print(f"✅ {len(CITY_PROFILES)} villes marocaines disponibles :")
for city in CITY_PROFILES.keys():
    print(f"   - {city}")
print("=" * 60)