import unicodedata
import string


# 1) Fonction de nettoyage du texte
def nettoyer_texte(t: str) -> str:
    """
    Objectif : rendre le texte plus stable pour la vectorisation.
    """
    if not isinstance(t, str):
        t = str(t)

    t = t.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())

    return t
