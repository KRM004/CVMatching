"""CV Matching & Job Recommendation — package partagé (pipeline, API, notebooks, tests).

Organisation :
  data/           catalogue d'offres (Rekrute) + données fictives
  preprocessing/  PDF -> texte, masquage PII, nettoyage, compétences
  ranking/        TF-IDF (baseline) · Sentence-BERT · embeddings · factory
  explain/        règles (L0) · Gemini + garde-fou (L1/A2)
  evaluation/     ground truth, métriques, ablations, rapport
  pipeline/       étapes hors-ligne + orchestrateur
  service.py      inférence (utilisée par l'API)
  api/            FastAPI
  cli.py          commande `cvmatch`
"""

__version__ = "1.0.0"
