# train_intents.py
from __future__ import annotations

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

from nlp_utils.text_cleaning import nettoyer_texte


FRENCH_STOPWORDS = {
    "a", "à", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "eux", "il", "je", "la", "le", "les", "leur", "lui",
    "ma", "mais", "me", "même", "mes", "moi", "mon", "ne", "nos", "notre",
    "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se",
    "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une",
    "vos", "votre", "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y"
}


def safe_clean(text: str) -> str:
    raw = (text or "").strip().lower()
    cleaned = nettoyer_texte(text)
    if cleaned is None:
        cleaned = ""
    cleaned = str(cleaned).strip()
    return cleaned if cleaned else raw


def read_intents_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8", sep=",", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", sep=",", on_bad_lines="skip")

    if len(df.columns) == 1 and ";" in df.columns[0]:
        try:
            df = pd.read_csv(path, encoding="utf-8", sep=";", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", sep=";", on_bad_lines="skip")

    return df


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"texte", "intention"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes {missing}. Colonnes: {list(df.columns)}")

    df = df.dropna(subset=["texte", "intention"]).copy()
    df["texte"] = df["texte"].astype(str).str.strip()
    df["intention"] = df["intention"].astype(str).str.strip()
    df = df[(df["texte"] != "") & (df["intention"] != "")].copy()

    df["texte_clean"] = df["texte"].apply(safe_clean)
    df = df.drop_duplicates(subset=["texte_clean", "intention"]).copy()

    return df


def main():
    csv_path = "intentions.csv"
    model_path = "model.joblib"

    df = read_intents_csv(csv_path)
    df = validate_and_prepare(df)

    print("✅ Répartition intentions :")
    print(df["intention"].value_counts().to_string())

    X = df["texte_clean"]
    y = df["intention"]

    counts = y.value_counts()
    stratify = y if (counts.min() >= 2) else None
    if stratify is None:
        print("\n⚠️ Certaines classes ont < 2 exemples -> stratify désactivé")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    # SVM linéaire + calibration => proba
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm, method="sigmoid", cv=3)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words=list(FRENCH_STOPWORDS),
        )),
        ("clf", clf)
    ])

    print("\n🔄 Entraînement...")
    pipe.fit(X_train, y_train)
    print("✅ Entraînement terminé")

    print("\n=== ÉVALUATION (TEST) ===")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipe, model_path)
    print(f"\n✅ Modèle sauvegardé dans '{model_path}'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
