# Harvestify

A Flask web application for agricultural decision support: crop recommendation, fertilizer suggestion, and (optionally) plant disease detection from leaf images.

## Overview

Harvestify is an end-to-end ML deployment project built while learning how to take trained models from notebook to a usable web app. It targets three problems farmers commonly face:

- **Crop recommendation** — given soil nutrient levels (N, P, K), soil pH, expected rainfall, and the current weather (temperature/humidity fetched live for a given city), recommend which crop is best suited to plant.
- **Fertilizer suggestion** — given a crop and its current N/P/K readings, compare them against ideal values for that crop (from a reference table) and suggest whether nitrogen, phosphorus, or potassium needs to be increased or decreased.
- **Plant disease detection** — classify a photo of a plant leaf into one of 38 disease/healthy classes (apple, corn, grape, potato, tomato, etc.) using a CNN. In this repository the disease-detection code path (PyTorch ResNet9 model loading and inference) is present but commented out in `app.py`, so only crop recommendation and fertilizer suggestion are active out of the box.

This project follows the well-known "Harvestify" tutorial pattern (crop/fertilizer/disease recommendation via Flask) used in several ML-deployment courses, adapted and run here as a personal learning project.

## Approach

- **Crop recommendation**: a scikit-learn `RandomForestClassifier`, trained offline and serialized to `models/RandomForest.pkl`, is loaded at app startup and used to predict a crop label from a 7-feature input vector `[N, P, K, temperature, humidity, ph, rainfall]`.
- **Fertilizer suggestion**: a rule-based lookup against `Data/fertilizer.csv` (ideal N/P/K per crop). The app computes the difference between the user's readings and the ideal values and returns the largest deviation (N, P, or K; high or low) with a corresponding textual recommendation from `utils/fertilizer.py`.
- **Weather lookup**: `weather_fetch()` calls the OpenWeatherMap API to get live temperature and humidity for the city entered by the user, which feeds into the crop model.
- **Plant disease detection**: intended to use a pretrained ResNet9 (`models/plant_disease_model.pth`) with a torchvision transform pipeline; this path is currently disabled (commented out) in `app.py`.

## Tech Stack

- Python, Flask (web app and routing)
- scikit-learn (RandomForestClassifier for crop recommendation)
- pandas / numpy (data handling)
- PyTorch / torchvision (disease-detection model, present but not wired up)
- Pillow (image handling)
- requests (OpenWeatherMap API calls)
- gunicorn (production server, see `Procfile`)

## Results

No accuracy/metric values are printed or logged in this repository — the crop-recommendation model ships as a pre-trained `.pkl` file with no accompanying training notebook, so no quantitative results can be honestly reported here.

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

The app serves:
- `/` — home page
- `/crop-recommend` and `/crop-predict` — crop recommendation form and result
- `/fertilizer` and `/fertilizer-predict` — fertilizer suggestion form and result
- `/disease-predict` — disease detection page (inactive; underlying model loading is commented out)

Note: `config.py` contains a hardcoded OpenWeatherMap API key, which should be replaced with your own key (ideally via an environment variable) before deploying.
