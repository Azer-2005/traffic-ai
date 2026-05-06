# 🚦 TrafficAI – Maroc

**Prédiction du trafic routier pour 27 villes marocaines**  
Application web basée sur Flask et un modèle de *Machine Learning* (Gradient Boosting).  
Elle estime le nombre de véhicules par heure en fonction de la ville, de l’heure, du jour, de la météo, de la température et des jours fériés.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Fonctionnalités

- ✅ Prédiction en temps réel via un formulaire web
- ✅ Graphique d’évolution du trafic sur 24h
- ✅ Heatmap horaire interactive
- ✅ API REST (prédiction unique et prévision 24h)
- ✅ Interface responsive et design moderne

---

## 🇲🇦 Villes couvertes

Casablanca, Rabat, Tanger, Fès, Marrakech, Agadir, Meknès, Oujda, Kénitra, Tétouan, Safi, El Jadida, Nador, Settat, Berrechid, Khouribga, Béni Mellal, Essaouira, Larache, Al Hoceïma, Laâyoune, Dakhla, Guelmim, Tarfaya, Chefchaouen, Ouzoud, Ifrane.

---

## 📊 Performances du modèle

| Métrique | Valeur |
|----------|--------|
| **MAE** (erreur moyenne absolue) | 36.3 véhicules/heure |
| **R²** (qualité de prédiction) | 0.929 (92,9%) |

---

## 🛠️ Stack technique

- **Backend** : Flask (Python)
- **Machine Learning** : scikit-learn (GradientBoostingRegressor)
- **Frontend** : HTML5, CSS3, JavaScript, Chart.js
- **Sérialisation** : Pickle
- **Environnement** : Python 3.12 + venv

---

## 🚀 Installation et exécution

```bash
# 1. Cloner le dépôt
git clone https://github.com/Azer-2005/trafficai-maroc.git
cd trafficai-maroc

# 2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) Réentraîner le modèle
python train.py

# 5. Lancer l’application
python app.py

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
