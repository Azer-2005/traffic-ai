# 🚦 TrafficAI — Prédiction de Trafic Routier

Application Flask professionnelle de prédiction du volume de trafic basée sur un modèle ML (GradientBoosting).

## 🏗 Structure
```
traffic_project/
├── data/
│   └── traffic.csv          # Dataset d'entraînement
├── model/
│   └── model.pkl            # Modèle sauvegardé (généré par train.py)
├── templates/
│   └── index.html           # Dashboard UI
├── app.py                   # Serveur Flask
├── train.py                 # Script d'entraînement
├── requirements.txt         # Dépendances
└── README.md
```

## 🚀 Installation & Lancement

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle
python train.py

# 4. Lancer l'application
python app.py
```

Ouvre **http://127.0.0.1:5000** dans ton navigateur.

## 🔌 API

| Endpoint        | Méthode | Description                        |
|-----------------|---------|------------------------------------|
| `/`             | GET     | Dashboard principal                |
| `/predict`      | POST    | Prédiction pour une heure donnée   |
| `/api/hourly`   | POST    | Prévision complète sur 24h (JSON)  |

## 🤖 Modèle ML

- **Algorithme** : GradientBoostingRegressor
- **Features** : heure, jour, météo, température, jour férié, heure de pointe, weekend, encodage cyclique
- **Cible** : volume de véhicules / heure


