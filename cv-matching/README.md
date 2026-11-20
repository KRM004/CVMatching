# Intelligent CV Matching & Job Recommendation System

CV PDF → extraction du texte → masquage des données personnelles → **Top 3 offres** (TF-IDF ou
Sentence-BERT) → **explications Gemini** (compétences en commun / à développer, preuves) ou à règles.
Catalogue d'offres : **Rekrute (Maroc)** — `adlbh/rekrute-2005-2022` sur Hugging Face.

> Le « matching score » est une **proximité textuelle/sémantique**, pas une probabilité d'embauche.
> Aucun résultat chiffré n'est fourni dans ce dépôt : tout est **mesuré par vos scripts** sur votre jeu annoté.

---------------------------------------------------------------------------------------------------

## 1. Installation (VS Code)

Prérequis : Python 3.10 – 3.12, Git, VS Code (extensions Python + Jupyter ; Docker optionnel).
Ouvrez le dossier (`File → Open Folder`), VS Code propose d'installer les extensions recommandées.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # si bloqué : Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
```

Puis `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`. Éditez `.env` et mettez votre
`GEMINI_API_KEY` (clé gratuite : Google AI Studio). **Ne committez jamais `.env`** (déjà dans `.gitignore`).

Test rapide de l'installation : `pytest` (ou tâche VS Code **Tests**).

## 2. Pipeline, étape par étape

| # | Commande | Rôle | Livrable |
|---|----------|------|----------|
| 1 | `python scripts/prepare_data.py` | Télécharge Rekrute, inspecte, nettoie, dédoublonne | `data/processed/jobs_clean.parquet` |
| 2 | `python scripts/build_artifacts.py` | Embeddings SBERT des offres (variantes *clean* et *raw*) | `artifacts/emb_*.npy` |
| 3 | `python scripts/make_demo_cvs.py` | 20 CV **fictifs** (txt + pdf) pour tester | `data/eval/cvs/` |
| 4 | `python scripts/make_eval_pool.py` | Pool d'offres candidates à annoter par CV | `data/eval/pool_to_annotate.csv` |
| 5 | *Annotation à deux* (voir §3) | Ground truth | colonnes `annotator_1/2` |
| 6 | `python scripts/finalize_labels.py` | Fusion + accord/kappa | `data/eval/labels.csv` |
| 7 | `python scripts/run_ablation.py` | B0 / F1 / A1 : Precision@3 + latence | `results/ablation_ml*.csv` |
| 8 | `python scripts/run_llm_ablation.py --no-cache` | L0 / L1 / A2 : fidélité + latence | `results/ablation_llm.csv` |

Options utiles : `prepare_data.py --sectors informatique ingenieur --max-rows 8000` (sous-ensemble IT) ·
`prepare_data.py --raw chemin.parquet` (fichier local) · `make_eval_pool.py --no-sbert` (sans modèle).
Les tâches VS Code (`Ctrl+Shift+P` → *Tasks: Run Task*) lancent les mêmes commandes.

**CV de test réalistes.** Les 20 CV de l'étape 3 servent à vérifier la chaîne technique. Pour l'évaluation,
remplacez-les par des CV **anonymisés/consentis** (par ex. un échantillon du dataset Figshare « GIS + NLP job
matching », CV emploi.ma anonymisés — vérifiez sa licence et son schéma) convertis en `.txt` ou `.pdf`
dans `data/eval/cvs/`, puis relancez les étapes 4 à 8.

## 2 bis. Grand corpus de CV (ex. 5054 CV)

```bash
# Dossier de PDF/TXT (sous-dossier = catégorie) OU CSV/Parquet/Excel (colonnes détectées automatiquement)
python scripts/import_cvs.py --source data/raw/cvs/ --sample-eval 60
python scripts/import_cvs.py --source data/raw/Resume.csv --text-col Resume_str --category-col Category --sample-eval 60
python scripts/batch_match.py --ranker tfidf      # puis --ranker sbert
```

- `import_cvs.py` : lit, **masque les PII dès l'import**, retire les CV vides/trop courts et les doublons,
  écrit `data/processed/cvs.parquet`. `--sample-eval N` copie N CV (stratifiés par catégorie) dans
  `data/eval/cvs/` : ce sont **eux** qu'on annote (étapes 4-6) — annoter 5054 CV à la main est irréaliste.
- `batch_match.py` : Top 3 de **tous** les CV par lots (sans Gemini, quota), et `results/batch_*.csv` :
  latence par CV, distribution des scores, couverture du catalogue, secteur recommandé par catégorie de CV.
- Vérifiez la **licence** du dataset de CV et citez-la dans le rapport ; ne committez jamais de vrais CV.

## 3. Ground truth (Precision@3)

1. `pool_to_annotate.csv` contient, pour chaque CV : top-6 TF-IDF ∪ top-6 SBERT ∪ 4 offres aléatoires,
   mélangées (les annotateurs ne voient pas le rang).
2. Chaque membre remplit **seul** sa colonne (`1` = pertinente, `0` = non) avec une grille écrite
   (ex. : métier visé + compétences clés compatibles + niveau d'études cohérent).
3. `finalize_labels.py` : pertinent = **les deux** ont mis 1 ; il affiche accord et kappa de Cohen et liste
   les désaccords (`data/eval/disagreements.csv`) à discuter.
4. **Figez `labels.csv` avant les derniers tests** ; ne réglez aucun hyperparamètre dessus.
5. Limites à citer : petit échantillon, biais de pooling (le pool vient des deux modèles), catalogue 2019-2022.
6. `--proxy` (labels automatiques par mots-clés) sert uniquement à tester la chaîne : **jamais** pour le rapport.

## 4. Ablations (même CV, même pool, un seul facteur change)

| Test | Configuration | Commande / réglage |
|------|---------------|--------------------|
| B0 | TF-IDF + cosine | `run_ablation.py` |
| F1 | SBERT + cleaning | `run_ablation.py` |
| A1 | SBERT **sans** cleaning | `run_ablation.py` |
| L0 | Top 3 + explication à règles | `run_llm_ablation.py` |
| L1 | Gemini avec descriptions d'offres | `run_llm_ablation.py` |
| A2 | Gemini **sans** descriptions d'offres | `run_llm_ablation.py` |

`delta = version complète − version ablatée` (ex. 0,80 − 0,74 = +0,06). Delta nul ou négatif ⇒ pas de
bénéfice démontré **sur ce test**. Gemini est jugé sur la **fidélité** (affirmations non retrouvées dans les
sources, comptées automatiquement + grille manuelle `manual_*` à remplir à deux puis
`run_llm_ablation.py --summarize`), pas sur la Precision@3. Notebooks : `05_Ablation_ML`, `06_Evaluation_Ablation`.

## 5. Lancer l'application

**Développement (F5)** : onglet *Run and Debug* → **Full stack (API + UI)** (FastAPI :8000 + Streamlit :8501).
Ou à la main, dans deux terminaux :
```bash
uvicorn backend.main:app --reload --port 8000      # docs : http://localhost:8000/docs
streamlit run frontend/app.py                      # UI   : http://localhost:8501
```
**Docker** (après les étapes 1 et 2) :
```bash
docker compose up --build
```
Deux conteneurs (backend + frontend), données et embeddings montés en **lecture seule**. Si Docker pose problème :
plan B = notebook `00_Colab_Run_All.ipynb` (exécutable de haut en bas sur Colab, livrable exigé par l'UIR).

Sans clé Gemini ou en cas de quota dépassé, l'API affiche quand même le Top 3 avec l'explication à règles
(`source: "rules_fallback"` + avertissement). Les appels Gemini sont mis en cache (`data/cache/`).

## 6. Configuration (`.env`)

| Variable | Valeurs | Effet |
|----------|---------|-------|
| `RANKER` | `sbert` / `tfidf` | modèle de classement |
| `CLEANING` | `on` / `off` | nettoyage du texte (A1) |
| `EXPLAINER` | `gemini` / `rules` | explications par défaut |
| `JOB_CONTEXT` | `on` / `off` | Gemini avec/sans description d'offre (A2) |
| `LANGUAGE` | `fr` / `en` | langue des explications |
| `SBERT_MODEL` | nom Hugging Face | défaut : `paraphrase-multilingual-MiniLM-L12-v2` |
| `GEMINI_MODEL` | nom du modèle | à ajuster si votre quota/clé n'y donne pas accès |
| `GEMINI_MIN_INTERVAL` | secondes | espacement entre appels (quota gratuit) |

Changer `CLEANING` ou `SBERT_MODEL` impose de relancer `build_artifacts.py` (embeddings dédiés par variante).

## 7. Structure du dépôt

```
src/cvmatch/        package métier (importé par notebooks, API, scripts, tests)
  pdf_extractor.py    extract_pdf_text(file)                         [M1.3]
  cleaning.py / pii.py / skills.py   nettoyage (garde C++, C#, .NET…), masquage PII, compétences
  ranking/            tfidf.py (baseline) · sbert.py · base.py (top-k, diversité)   [M1.4-M1.6]
  matching.py         recommend_jobs(resume_text, jobs, top_k=3)    [M1.7]
  explain/            rules.py (L0) · gemini.py (L1/A2, cache, garde-fou) · prompts.py   [M2.3-M2.4]
  service.py          pipeline extract → match → explain
  schemas.py          contrat JSON partagé (Pydantic)               [M2.1]
  evaluation.py       Precision@3, latence, deltas, labels
  cv_corpus.py        import d'un grand corpus de CV (dossier ou CSV), masquage, dédoublonnage
backend/  main.py (POST /match, /match_text, /health) + Dockerfile    [M2.5, M2.7]
frontend/ app.py (Streamlit) + Dockerfile                              [M2.6, M2.7]
scripts/  prepare_data · build_artifacts · import_cvs · batch_match · make_* · run_ablation · run_llm_ablation
notebooks/ 00_Colab_Run_All · 01_EDA · 02_TFIDF_Baseline · 03_SentenceBERT · 05_Ablation_ML · 06_Evaluation_Ablation
tests/ · compose.yaml · .env.example · .vscode/
```

Correspondance avec le guide : `ml/pdf_extractor.py` → `src/cvmatch/pdf_extractor.py` ;
`ml/matching.py` → `src/cvmatch/matching.py` + `ranking/` ; `llm/gemini_client.py` → `src/cvmatch/explain/gemini.py`.

## 8. Choix de conception à justifier à l'oral

- **Rankers interchangeables** (même interface) : l'ablation ne change qu'une option, et les notebooks
  exécutent exactement le code de l'API.
- **Évaluation sur pool figé** : chaque CV est classé parmi *ses* offres annotées ; une paire non annotée n'est
  jamais traitée comme négative.
- **Embeddings pré-calculés** ; seul le CV est encodé à la requête (les longs CV sont découpés en blocs puis moyennés).
- **Diversité du Top 3** (`diversify`) : évite trois fois le même intitulé publié par plusieurs agences.
- **Garde-fou anti-invention** : toute compétence / preuve renvoyée par Gemini est vérifiée dans le CV et
  l'offre ; ce qui n'y figure pas est retiré et compté (`unverified_skills`, `guard_removed`).
- **Vie privée** : e-mail, téléphone, liens et adresse masqués **avant** logs et appel Gemini ; utilisez des CV
  fictifs/anonymisés/consentis uniquement.
- **Limites connues** : catalogue 2005-2022 (récent = 2019-2022 par défaut) ; pas de filtre d'éligibilité
  (diplôme/expérience) ; dictionnaire de compétences limité (extensible dans `skills.py`) ; le modèle SBERT tronque
  les très longues offres.

## 9. État de validation

Exécuté et vérifié hors-ligne sur des **données synthétiques** : nettoyage, masquage PII, extraction de
compétences, TF-IDF, chaîne SBERT (encodeur factice), garde-fou Gemini (client simulé), service, métriques,
`finalize_labels`, scripts 1/3/4/7/8. **Non exécuté dans l'environnement de conception** (pas d'accès réseau) :
téléchargement Hugging Face, modèle SBERT réel, appels Gemini réels, PyMuPDF, FastAPI/Streamlit, Docker. Les tests
`tests/test_pdf_api.py` couvrent PDF + API : lancez `pytest` après installation.

## 10. Dépannage

- `FileNotFoundError … jobs_clean` → étape 1 ; `Embeddings introuvables` → étape 2.
- `/health` affiche `degraded` → lisez le champ `error` (artefacts manquants).
- Gemini 429 / quota → augmentez `GEMINI_MIN_INTERVAL`, gardez le cache, ou `EXPLAINER=rules`.
- PDF « scanné » (aucun texte) → utilisez un PDF numérique (erreur 422 attendue).
- Noms de colonnes différents du dataset réel → ajustez les constantes en tête de `src/cvmatch/data.py`.
