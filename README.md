# Chest X-ray Classifier (COVID-19 / Pneumonia / Normal)

A Streamlit app that serves a CNN trained on Kaggle (`best_model.keras`) to classify chest X-ray images.

> ⚠️ Educational demo only – not a medical device and not for diagnosis.

## Project flow

```
Kaggle notebook ──► best_model.keras + class_names.json ──► Kaggle API token ──► Streamlit app ──► Prediction
(training)          (notebook output)                        (.env)              (app.py)
```

1. **Data** – COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset (6,902 images, 3 classes: `covid`, `normal`, `pneumonia`). 80/20 train/validation split (seed 42), images resized to 224×224, pixels scaled to `[0, 1]`.
2. **Training** – Kaggle notebook [cnn-realdata-covid19](https://www.kaggle.com/code/olaelshiekh/cnn-realdata-covid19):
   - Data augmentation (random flip, rotation, zoom)
   - 3 × (Conv2D + MaxPooling) → Flatten → Dense 128/64/32 with Dropout 0.5 → Softmax(3)
   - Adam, sparse categorical cross-entropy, EarlyStopping / ReduceLROnPlateau / ModelCheckpoint on `val_loss`
   - Validation accuracy ≈ **89%** (F1: covid 0.91, normal 0.87, pneumonia 0.90)
3. **Export** – the best checkpoint (`best_model.keras`) and `class_names.json` are saved as the notebook output.
4. **Connect to Kaggle** – the Kaggle API token is stored in `.env` as `KAGGLE_API_TOKEN`. [kaggle_sync.py](kaggle_sync.py) loads it with `python-dotenv` and downloads the notebook output into `model/` (only if it is not already there).
5. **Serve** – [app.py](app.py) loads the model once (`st.cache_resource`), then for each uploaded image:
   convert to RGB → resize to 224×224 → divide by 255 → `model.predict` → show the top class, confidence and per-class probabilities.

## UI

- Sidebar: model info, probability-chart toggle, low-confidence threshold, medical disclaimer
- Left: image upload (PNG/JPG) or camera input with preview
- Right: colour-coded predicted class, confidence, per-class metrics and bar chart, low-confidence warning

## Project structure

```
DL_Deployment/
├── app.py            # Streamlit UI + inference
├── kaggle_sync.py    # Downloads the model from Kaggle using the API token
├── model/            # best_model.keras, class_names.json (downloaded)
├── notebook/         # Copy of the training notebook
├── requirments.txt   # Pinned dependencies (ML + TensorFlow/Keras + Kaggle)
├── .env              # KAGGLE_API_TOKEN (not committed)
└── README.md
```

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirments.txt

# Get a token at https://www.kaggle.com/settings (API section) and put it in .env:
echo 'KAGGLE_API_TOKEN=<your-token>' > .env

python kaggle_sync.py        # optional: pre-download the model (app.py does it automatically)
streamlit run app.py
```

Open http://localhost:8501.

## Notes

- The training pipeline applied `Rescaling(1/255)` outside the model, so the app must (and does) divide inputs by 255.
- Never commit `.env`; it is listed in `.gitignore`.
