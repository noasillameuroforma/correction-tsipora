Chatbot – Projet corrigé (Back-end + front de test)

Ce projet contient :

un back-end FastAPI pour la détection d’intentions (NLP)

un petit front très simple ajouté uniquement pour tester rapidement le chatbot (formulaire + affichage de la réponse)

Prérequis


pip

(optionnel mais conseillé) virtualenv

Vérifier les versions :

python --version
pip --version

Structure du projet
chatbot/
├── back-end/
│   ├── train_intents.py
│   ├── serve.py
│   ├── intentions.csv
│   ├── model.joblib
│   ├── requirements.txt
│   └── nlp_utils/
│       └── text_cleaning.py
│
├── front-test/
│   └── index.html
│
└── README.md

Installation du back-end
1Créer un environnement virtuel (recommandé)

Depuis le dossier back-end :

python -m venv .venv


Activer l’environnement :

macOS / Linux

source .venv/bin/activate


Windows

.venv\Scripts\activate

Installer les dépendances
pip install -r requirements.txt


Entraîner le modèle

Toujours dans le dossier back-end :

python train_intents.py


Cela va :

charger intentions.csv

entraîner le modèle NLP

générer le fichier model.joblib

Lancer l’API
uvicorn serve:app --reload --host 0.0.0.0 --port 8000


Si tout est OK, tu verras :

Uvicorn running on http://0.0.0.0:8000

Utiliser le petit front de test

Le dossier front-test contient un HTML très simple pour tester le chatbot.

Lancer le back-end

(voir section précédente)

Ouvrir le front

Ouvre simplement front-test/index.html dans ton navigateur

Tape un message

La requête est envoyée à l’API FastAPI

La réponse, l’intention et la confidence s’affichent

le front sert uniquement au test, il n’est pas destiné à la production.

Notes importantes

Le modèle est volontairement simple, adapté à un petit dataset

Les probabilités peuvent sembler basses (0.15 – 0.35), c’est normal avec peu de données

Le seuil de confiance est réglé bas pour éviter les faux fallbacks

Pour améliorer les résultats :

ajouter plus d’exemples dans intentions.csv

ajouter des variantes proches du langage utilisateur réel