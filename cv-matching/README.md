# Intelligent CV Matching & Job Recommendation System

CV PDF → extraction du texte → masquage des données personnelles → **Top 3 offres** (TF-IDF ou
Sentence-BERT) → **explications** (Gemini avec garde-fou anti-invention, ou à règles).
Catalogue d'offres : **Rekrute (Maroc)** — `adlbh/rekrute-2005-2022` (Hugging Face).

> Le « matching score » est une **proximité textuelle/sémantique**, pas une probabilité d'embauche.
> Aucun résultat chiffré n'est fourni : tout est **mesuré par le pipeline** sur votre jeu annoté.

---

## 1. Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .\.venv\Scripts\Activate.ps1
pip install -e ".[all]"                                # ou ".[dev,ui]" sans SBERT/torch
cp .env.example .env                                   # puis GEMINI_API_KEY (optionnel)

cvmatch pipeline --offline     # démo 100 % hors-ligne (≈ 2 s) : vérifie toute la chaîne
cvmatch pipeline               # vrai pipeline : Rekrute + Sentence-BERT (+ Gemini si clé)
cvmatch serve                  # API  -> http://localhost:8000/docs
streamlit run ui/app.py        # UI   -> http://localhost:8501
```

`make help` liste les mêmes raccourcis (`make demo`, `make test`, `make zip`, …).

## 2. Le pipeline

```
 ┌─────────┐   ┌───────┐   ┌─────┐   ┌──────┐   ┌────────┐   ┌───────────┐   ┌────────────┐   ┌────────┐
 │ prepare │ → │ embed │ → │ cvs │ → │ pool │ → │ labels │ → │ ablate-ml │ → │ ablate-llm │ → │ report │
 └─────────┘   └───────┘   └─────┘   └──────┘   └────────┘   └───────────┘   └────────────┘   └────────┘
```

| Étape | Rôle | Sorties |
|-------|------|---------|
| `prepare` | Télécharge Rekrute (ou fichier local / données fictives), filtre, nettoie, dédoublonne, valide | `data/processed/jobs_clean.parquet` |
| `embed` | Embeddings SBERT des offres, variantes *clean* (F1) et *raw* (A1) | `artifacts/emb_<modèle>_{clean,raw}.npy` |
| `cvs` | CV de test : les vôtres dans `data/eval/cvs/`, sinon 20 CV **fictifs** (.txt + .pdf) | `data/eval/cvs/`, `cv_meta.json` |
| `pool` | Par CV : top-N TF-IDF ∪ top-N SBERT ∪ offres aléatoires, mélangés | `data/eval/pool_to_annotate.csv`, `proxy_labels.csv` |
| `labels` | Fusionne les 2 annotateurs (accord, kappa) ; sinon bascule en labels **proxy** | `data/eval/labels.csv`, `disagreements.csv` |
| `ablate-ml` | B0 TF-IDF / F1 SBERT+cleaning / A1 SBERT sans cleaning : Precision@k + latence | `results/ablation_ml[_PROXY].csv`, `…_summary…` |
| `ablate-llm` | L0 règles / L1 Gemini / A2 Gemini sans description : fidélité + latence | `results/ablation_llm.csv`, `…_summary.csv` |
| `report` | Rapport Markdown consolidé | `results/report.md` |

Chaque exécution écrit `results/pipeline_run.json` (config, durée et statut de chaque étape).

```bash
cvmatch pipeline --from pool          # reprend à une étape
cvmatch pipeline --to embed           # s'arrête après une étape (ex. avant Docker)
cvmatch pipeline --only ablate-ml report
cvmatch embed --force                 # refait une étape déjà faite
cvmatch pipeline --tfidf-only         # sans Sentence-BERT
cvmatch prepare --sectors informatique ingenieur --max-rows 8000
cvmatch prepare --raw chemin/rekrute.parquet
```

Les étapes coûteuses (`prepare`, `embed`, `pool`) sont **sautées si leurs sorties existent**
(`--force` pour refaire). Un pool **déjà annoté n'est jamais écrasé**.

Paramètres : `configs/pipeline.yaml` (source des données, années, échantillon, variantes
d'embeddings, taille du pool, k, labels, ablation LLM). Les options CLI priment.

## 3. Ground truth (Precision@3)

1. `pool_to_annotate.csv` : pour chaque CV, top-6 TF-IDF ∪ top-6 SBERT ∪ 4 offres aléatoires, mélangées.
2. Chaque membre remplit **seul** sa colonne (`annotator_1` / `annotator_2` : 1 = pertinente, 0 = non)
   avec une grille écrite (métier visé + compétences clés + niveau d'études cohérent).
3. `cvmatch labels` : pertinent = **les deux** ont mis 1 ; accord + kappa de Cohen, désaccords listés.
4. **Figez `labels.csv` avant les derniers tests** ; aucun hyperparamètre réglé dessus.
5. Limites : petit échantillon, biais de pooling, catalogue 2019-2022.
6. Labels **proxy** (mots-clés automatiques) = test technique uniquement : signalés `_PROXY` partout.

## 4. Ablations (même CV, même pool, un seul facteur change)

| Test | Configuration | Mesure |
|------|---------------|--------|
| B0 | TF-IDF + cosinus | Precision@3, latence |
| F1 | SBERT + cleaning | Precision@3, latence |
| A1 | SBERT **sans** cleaning | Precision@3, latence |
| L0 | Top 3 + explication à règles | fidélité, latence |
| L1 | Gemini avec descriptions d'offres | fidélité, latence |
| A2 | Gemini **sans** descriptions d'offres | fidélité, latence |

`delta = version complète − version ablatée`. Delta ≤ 0 ⇒ pas de bénéfice démontré **sur ce test**.
Gemini est jugé sur la **fidélité** (éléments retirés par le garde-fou + grille manuelle `manual_*`
remplie à deux, puis `cvmatch ablate-llm --summarize`).

## 5. Application

| Mode | Commande |
|------|----------|
| API | `cvmatch serve` (ou `uvicorn cvmatch.api.app:app --reload`) |
| UI | `streamlit run ui/app.py` (variable `API_URL`, défaut `http://localhost:8000`) |
| VS Code | *Run and Debug* → **Full stack (API + UI)** ; *Tasks* pour chaque étape |
| Docker | `cvmatch pipeline --to embed` puis `docker compose up --build` |
| CLI | `cvmatch match data/eval/cvs/cv_01.pdf --explainer rules [--json]` |

Endpoints : `GET /health`, `POST /match` (PDF, multipart), `POST /match_text` (JSON).
Sans clé Gemini ou quota dépassé : Top 3 + explication à règles (`source: "rules_fallback"` + avertissement).

## 6. Configuration du service (`.env`)

| Variable | Valeurs | Effet |
|----------|---------|-------|
| `RANKER` | `sbert` / `tfidf` | modèle de classement |
| `CLEANING` | `on` / `off` | nettoyage du texte |
| `EXPLAINER` | `gemini` / `rules` | explications par défaut |
| `JOB_CONTEXT` | `on` / `off` | Gemini avec/sans description d'offre |
| `LANGUAGE` | `fr` / `en` | langue des explications |
| `SBERT_MODEL` | nom Hugging Face / `hashing` | `hashing` = encodeur factice hors-ligne (tests) |
| `GEMINI_MODEL`, `GEMINI_MIN_INTERVAL` | | modèle Gemini, espacement des appels |
| `JOBS_PATH`, `ARTIFACTS_DIR`, `CACHE_DIR`, `EVAL_DIR`, `RESULTS_DIR` | chemins | emplacements |

`cvmatch info` affiche la configuration effective et l'état des fichiers.

## 7. Structure

```
cv-matching/
├── configs/pipeline.yaml        paramètres des étapes hors-ligne
├── src/cvmatch/
│   ├── cli.py                   commande `cvmatch` (pipeline, étapes, match, serve, info, package)
│   ├── config.py                Settings (.env)
│   ├── schemas.py               contrat JSON partagé (Pydantic)
│   ├── service.py               inférence : CV -> PII -> nettoyage -> ranking -> explications
│   ├── matching.py              recommend_jobs(resume_text, jobs, top_k=3)
│   ├── data/                    catalog.py (Rekrute) · synthetic.py (données fictives)
│   ├── preprocessing/           pdf.py · pii.py · cleaning.py · skills.py
│   ├── ranking/                 base.py · tfidf.py · sbert.py · embeddings.py · factory.py
│   ├── explain/                 rules.py · gemini.py (garde-fou, cache) · prompts.py · factory.py
│   ├── evaluation/              annotation.py · metrics.py · ablation_ml.py · ablation_llm.py · report.py
│   ├── pipeline/                config.py · steps.py · runner.py
│   ├── api/app.py               FastAPI
│   └── utils/                   io.py · cache.py · logging.py · packaging.py
├── ui/app.py                    Streamlit (client HTTP de l'API)
├── notebooks/                   00 Colab tout-en-un · 01 EDA · 02 TF-IDF · 03 SBERT · 05/06 ablations
├── tests/unit/ · tests/integration/   (dont le pipeline complet hors-ligne)
├── docker/ · compose.yaml · Makefile · .env.example · .vscode/
├── data/{raw,processed,cache,eval}/  artifacts/  results/   (générés, non versionnés)
```

## 8. Choix de conception

- **Une seule source de vérité** : notebooks, CLI, API et tests appellent le même package.
- **Pipeline par étapes** avec entrées/sorties explicites, reprise, cache et manifeste d'exécution.
- **Rankers / explainers interchangeables** (factories) : une ablation = un seul paramètre qui change.
- **Évaluation sur pool figé** : une paire non annotée n'est jamais traitée comme négative.
- **Embeddings pré-calculés** ; seul le CV est encodé à la requête (découpé en blocs puis moyenné).
- **Diversité du Top 3** : pas trois fois le même intitulé publié par plusieurs agences.
- **Garde-fou anti-invention** : toute compétence/preuve de Gemini est vérifiée dans le CV et l'offre.
- **Vie privée** : e-mail, téléphone, liens, adresse masqués **avant** logs et appel Gemini.
- **Limites** : catalogue 2005-2022, pas de filtre d'éligibilité, dictionnaire de compétences extensible
  (`preprocessing/skills.py`), troncature des très longues offres par SBERT.

## 9. Tests & archive

```bash
pytest                 # unitaires + intégration (pipeline hors-ligne, API, PDF, CLI)
cvmatch package        # dist/cv-matching.zip (code + config + docs, sans données ni .env)
```

## 10. Dépannage

- `Catalogue introuvable` → `cvmatch prepare` ; `Embeddings introuvables` → `cvmatch embed`.
- `/health` = `degraded` → lire le champ `error`.
- Gemini 429 / quota → augmenter `GEMINI_MIN_INTERVAL`, garder le cache, ou `EXPLAINER=rules`.
- PDF scanné (aucun texte) → PDF numérique requis (erreur 422).
- Colonnes différentes dans le dataset réel → constantes en tête de `src/cvmatch/data/catalog.py`.
