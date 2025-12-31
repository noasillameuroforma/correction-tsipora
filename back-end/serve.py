# serve.py
from __future__ import annotations

import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from nlp_utils.text_cleaning import nettoyer_texte


MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.10"))

RESPONSES = {
    "salutation": "Bonjour ! Comment puis-je vous aider ?",
    "au_revoir": "Au revoir ! À bientôt 🙂",
    "presentation": "Je suis l’assistant du site d’échange de services de la ville de Sarcelles.",
    "inscription": "Vous pouvez vous inscrire via le formulaire d’inscription disponible sur le site.",
    "connexion": "Pour vous connecter, cliquez sur “Connexion” puis entrez vos identifiants.",
    "deconnexion": "Pour vous déconnecter, cliquez sur “Déconnexion” depuis votre compte.",
    "modifier_compte": "Vous pouvez modifier votre profil depuis votre espace personnel.",
    "supprimer_compte": "Vous pouvez supprimer votre compte depuis les paramètres du profil.",
    "creer_service": "Pour proposer un service, allez dans “Créer un service”.",
    "supprimer_service": "Pour supprimer un service, ouvrez votre annonce puis “Supprimer”.",
    "voir_services": "Vous pouvez consulter vos services dans “Mes services”.",
    "rechercher_service": "Vous pouvez rechercher un service via la recherche ou le catalogue.",
    "reserver_service": "Pour réserver, ouvrez un service puis cliquez sur “Réserver”.",
    "catalogue": "Vous trouverez la liste des services dans la section “Catalogue”.",
    "accueil": "Pour revenir à l’accueil, cliquez sur “Accueil” ou sur le logo.",
}

FALLBACK_MESSAGE = "Je ne suis pas sûr d’avoir compris votre demande. Pouvez-vous reformuler ?"


class Query(BaseModel):
    text: str


app = FastAPI(title="Intent Classifier API", version="1.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_clean(text: str) -> str:
    raw = (text or "").strip().lower()
    cleaned = nettoyer_texte(text)
    if cleaned is None:
        cleaned = ""
    cleaned = str(cleaned).strip()
    return cleaned if cleaned else raw


def get_clf(model):
    return model.named_steps["clf"] if hasattr(model, "named_steps") and "clf" in model.named_steps else model


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}. Lance train_intents.py d'abord.")

print("🔄 Chargement du modèle...")
MODEL = joblib.load(MODEL_PATH)
print("✅ Modèle chargé.")


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "threshold": CONFIDENCE_THRESHOLD}


@app.post("/predict")
def predict(q: Query):
    clean_text = safe_clean(q.text)

    predicted_intent = MODEL.predict([clean_text])[0]

    confidence = None
    debug = {
        "raw_text": q.text,
        "clean_text": clean_text,
        "predicted_intent": predicted_intent,
    }

    try:
        if hasattr(MODEL, "predict_proba"):
            probas = MODEL.predict_proba([clean_text])[0]
            clf = get_clf(MODEL)

            # Pour CalibratedClassifierCV, les classes sont sur clf.classes_
            if hasattr(clf, "classes_"):
                classes = list(clf.classes_)
                confidence = float(probas[classes.index(predicted_intent)]) if predicted_intent in classes else float(max(probas))
            else:
                confidence = float(max(probas))
    except Exception as e:
        debug["proba_error"] = str(e)
        confidence = None

    debug["confidence"] = confidence

    if confidence is None or confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "fallback",
            "response": FALLBACK_MESSAGE,
            **debug,
        }

    return {
        "intent": predicted_intent,
        "response": RESPONSES.get(predicted_intent, FALLBACK_MESSAGE),
        **debug,
    }
