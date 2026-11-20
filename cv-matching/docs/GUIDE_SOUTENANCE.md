# Guide : comprendre, tester et défendre le projet CVMatching

## 1. Le projet en 30 secondes (à savoir par cœur)

> « Un étudiant dépose son CV en PDF. Le système extrait le texte, masque ses données personnelles,
> le compare à ~15 000 offres marocaines (Rekrute 2019-2022) et renvoie les **3 offres les plus proches**,
> avec une **explication** : compétences en commun, compétences à développer, et des **preuves** copiées
> du CV ou de l'offre. Nous comparons deux méthodes de classement (TF-IDF et Sentence-BERT) et deux
> méthodes d'explication (règles et Gemini), en mesurant la contribution de chaque composant par ablation. »

### Le pipeline

```
CV.pdf ──► extraction (PyMuPDF) ──► masquage PII ──► nettoyage ──► classement ──► Top 3 ──► explication
          pdf_extractor.py        pii.py           cleaning.py    ranking/       diversify  explain/
                                                                  TF-IDF | SBERT            règles | Gemini
                                                                                            + garde-fou
```

| Fichier | Rôle | Ce qu'il faut savoir expliquer |
|---|---|---|
| `data.py` | Charge et nettoie les offres Rekrute | Filtre année ≥ 2019, supprime offres sans description, doublons exacts (hash MD5) |
| `pdf_extractor.py` | Texte du PDF | Un PDF scanné (image) n'a pas de texte → erreur 422 explicite |
| `pii.py` | Masque e-mail, téléphone (+212, 06…), LinkedIn, adresse | Fait **avant** tout log et tout appel à Gemini (RGPD / loi 09-08 au Maroc) |
| `cleaning.py` | Supprime HTML, URL, puces | Garde `C++`, `C#`, `.NET`, `Node.js` (un nettoyage naïf les détruirait) |
| `ranking/tfidf.py` | Baseline B0 | Pondère les mots rares ; uni+bigrammes ; similarité cosinus |
| `ranking/sbert.py` | Modèle F1 | Embeddings multilingues (FR/EN/AR) ; CV long découpé en blocs de 110 mots puis moyenné |
| `ranking/base.py` | Top-k + diversité | Évite 3 fois « Data Analyst (H/F) » publié par 3 agences |
| `skills.py` | Dictionnaire de ~100 compétences | Recherche sans accents ni casse ; `java` ≠ `javascript` |
| `explain/rules.py` | Explication L0 sans LLM | Compétences de l'offre ∩ compétences du CV |
| `explain/gemini.py` | Explication L1 (LLM) | Température 0, sortie JSON, cache, retries, **garde-fou** qui supprime toute compétence inventée |
| `service.py` | Orchestration | Même code utilisé par l'API, les notebooks et les scripts |
| `evaluation.py` | Precision@3, kappa de Cohen | Évaluation sur pool figé, double annotation |
| `cv_corpus.py` *(ajouté)* | Import d'un gros corpus de CV | 5054 CV : masquage, dédoublonnage, échantillon stratifié |
| `backend/main.py` | API FastAPI | `POST /match` (PDF), `POST /match_text`, `GET /health` |
| `frontend/app.py` | Interface Streamlit | Upload, Top 3, comparaison Gemini vs règles |

## 2. Ce qui a été testé (et le résultat)

| Test | Résultat |
|---|---|
| `pytest` (tests unitaires) | **27/27 passent** (22 au départ, dont 2 qui plantaient → corrigés, + 5 nouveaux) |
| Pipeline hors-ligne complet (étapes 1, 3, 4, 7, 8 sur données synthétiques) | OK |
| API réelle (`uvicorn`) : `/health` puis upload d'un PDF sur `/match` | OK, Top 3 + explications en ~16 ms (TF-IDF) |
| Import de 5054 CV (CSV simulé) | OK en 2,5 s |
| Matching en lot de 5054 CV (TF-IDF, 300 offres) | OK en 2,9 s, 0,6 ms/CV |
| Téléchargement Rekrute + modèle SBERT + Gemini réels | **Non testable ici** (réseau bloqué) → à lancer sur votre PC |

Les chiffres ci-dessus sont des **tests techniques sur données fictives** : ne les mettez jamais dans le rapport
comme résultats.

### Bugs trouvés et corrigés

1. **`tests/test_pdf_api.py` plantait** (`ModuleNotFoundError: backend`) : pytest ne trouvait pas le dossier
   `backend/`. Correction : `pythonpath = [".", "tests"]` dans `pyproject.toml`.
2. **`run_llm_ablation.py` ne nettoyait pas le CV** avant le classement, alors que l'API le fait : le Top 3
   évalué n'était donc pas le même que celui de l'application. Corrigé (même prétraitement que l'API).
3. **Top 3 peu diversifié** : « Cloud Engineer (H/F) » et « Cloud Engineer Junior » apparaissaient ensemble.
   La diversité ignore maintenant aussi junior / senior / confirmé / stagiaire.
4. **Échantillonnage stratifié** incompatible avec pandas ≥ 2.2 (colonne perdue) : corrigé et testé.
5. **`import fitz` déprécié** (avertissement à chaque PDF) → `import pymupdf`.

## 3. La structure est-elle « académique » ?

### Points forts (à mettre en avant)

- **Séparation claire** : `src/` (logique), `backend/` (API), `frontend/` (UI), `scripts/` (pipeline),
  `notebooks/` (exploration), `tests/`. Le même package est utilisé partout → reproductibilité.
- **Baseline + ablations** (B0/F1/A1, L0/L1/A2) : un seul facteur change à la fois. C'est la démarche
  scientifique attendue.
- **Ground truth honnête** : double annotation, kappa de Cohen, désaccords discutés, labels figés.
- **Honnêteté méthodologique** : le score n'est pas une probabilité d'embauche ; limites listées ; pas de
  chiffres inventés.
- **Éthique / vie privée** : masquage PII, CV jamais commités.
- **Ingénierie** : tests, Docker, configuration `.env`, cache, fallback si Gemini tombe.

### Faiblesses à corriger avant le rendu

| Problème | Pourquoi le jury peut le relever | Correction proposée |
|---|---|---|
| `README.md` à la racine quasi vide, et le commit s'appelle « Moroccan Public Jobs AI » | Première impression, incohérence de titre | Réécrire le README racine (titre, équipe, encadrant, lien vers `cv-matching/README.md`) |
| Versions des dépendances non figées | Reproductibilité | `pip freeze > requirements-lock.txt` une fois que tout marche |
| Une seule métrique (Precision@3) | Peu de recul sur le classement | Ajouter **MRR** et **nDCG@3** (quelques lignes dans `evaluation.py`) |
| Petit échantillon annoté | Différences peut-être dues au hasard | Donner un **intervalle de confiance bootstrap** ou un test de Wilcoxon apparié entre B0 et F1 |
| Notebooks numérotés 00, 01, 02, 03, 05, 06 (pas de 04) | Détail visible | Renommer ou ajouter un 04 (ex. analyse d'erreurs) |
| Pas de schéma d'architecture ni de dossier `docs/` | Attendu dans un rapport | Ce guide + un schéma (le pipeline ci-dessus) dans le rapport |
| Pas de licence ni de citation des datasets | Rigueur académique | Citer Rekrute (HF `adlbh/rekrute-2005-2022`) et le dataset de CV avec leurs licences |
| Pas d'intégration continue | Bonus « génie logiciel » | **Ajouté** : `.github/workflows/tests.yml` lance les tests à chaque push |

## 4. Insérer les 5054 CV

### Étape 1 : placer les fichiers (jamais dans Git)

- **Dossier de PDF/TXT/DOCX** → `cv-matching/data/raw/cvs/` (si vous avez des sous-dossiers par métier,
  ex. `cvs/INFORMATIQUE/…`, ils deviennent la catégorie).
- **Un fichier CSV** (ex. dataset Kaggle `Resume.csv` avec les colonnes `ID`, `Resume_str`, `Category`)
  → `cv-matching/data/raw/Resume.csv`.

### Étape 2 : importer

```bash
python scripts/import_cvs.py --source data/raw/Resume.csv --text-col Resume_str --category-col Category --sample-eval 60
```

ou, pour un dossier :

```bash
python scripts/import_cvs.py --source data/raw/cvs --sample-eval 60
```

Résultat : `data/processed/cvs.parquet` (texte **déjà masqué**) + 60 CV dans `data/eval/cvs/` à annoter.
Les colonnes sont détectées automatiquement ; `--text-col` n'est utile que si le nom est inhabituel.
Les `.docx` demandent `pip install python-docx`.

### Étape 3 : matching en lot

```bash
python scripts/batch_match.py --ranker tfidf
```

```bash
python scripts/batch_match.py --ranker sbert
```

Sorties dans `results/` : Top 3 de chaque CV, latence par CV, score moyen, couverture du catalogue, et
(si catégories) le secteur d'offres le plus recommandé par catégorie de CV.

### Pourquoi ne pas annoter les 5054 ?

5054 CV × 16 offres candidates ≈ 80 000 paires à juger à deux : impossible. **Méthode à présenter** :
- **60 CV** stratifiés, annotés à deux → Precision@3 / kappa (qualité, rigoureux) ;
- **5054 CV** → passage à l'échelle : latence, robustesse (aucun plantage), couverture, cohérence
  catégorie ↔ secteur (évaluation **indirecte**, à présenter comme telle).

### Attention à la langue

Le dataset Kaggle de CV est **en anglais** ; les offres Rekrute sont **en français**. TF-IDF compare des
mots : « developer » ≠ « développeur », il sera mauvais. Sentence-BERT **multilingue** rapproche les
deux langues : c'est un **excellent argument** pour justifier SBERT, à montrer avec les chiffres
`batch_summary_tfidf.csv` vs `batch_summary_sbert.csv`. Le dataset Figshare (CV emploi.ma, en français)
évite ce problème si vous voulez une évaluation « même langue ».

## 5. Installer, « compiler », lancer, tester (Windows)

Python ne se compile pas : on **installe** le package puis on **exécute**.

```powershell
cd cv-matching
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env      # puis mettre GEMINI_API_KEY dans .env
```

Tester :

```powershell
pytest -v
```

Pipeline réel, dans l'ordre :

```powershell
python scripts/prepare_data.py
python scripts/build_artifacts.py
python scripts/import_cvs.py --source data/raw/Resume.csv --sample-eval 60
python scripts/batch_match.py --ranker sbert
python scripts/make_eval_pool.py
# annoter data/eval/pool_to_annotate.csv à deux (annotator_1, annotator_2)
python scripts/finalize_labels.py
python scripts/run_ablation.py
python scripts/run_llm_ablation.py --no-cache
```

Application (deux terminaux) :

```powershell
uvicorn backend.main:app --reload --port 8000
```

```powershell
streamlit run frontend/app.py
```

Ouvrir http://localhost:8501 (UI) et http://localhost:8000/docs (API). Ou tout d'un coup : `docker compose up --build`.

Test rapide sans Internet (si le Wi-Fi de la salle de soutenance est mauvais) :

```powershell
$env:RANKER="tfidf"; $env:EXPLAINER="rules"
python scripts/prepare_data.py --synthetic
python scripts/make_demo_cvs.py
uvicorn backend.main:app --port 8000
```

## 6. Le jour de la soutenance

### Plan conseillé (15 min + questions)

1. **Problème** (1 min) : un jeune diplômé lit des dizaines d'offres à la main.
2. **Données** (2 min) : Rekrute (combien d'offres après nettoyage, quelles années), corpus de 5054 CV.
3. **Architecture** (2 min) : le schéma du pipeline de la section 1.
4. **Méthodes** (3 min) : TF-IDF (baseline) vs Sentence-BERT ; règles vs Gemini + garde-fou.
5. **Évaluation** (3 min) : protocole (pool, double annotation, kappa), tableau des ablations avec les deltas.
6. **Démo live** (3 min) : upload d'un CV fictif → Top 3 → cocher « Comparer Gemini et règles ».
7. **Limites et perspectives** (1 min).

### Préparer la démo

- Lancer l'API **avant** d'entrer (le chargement de SBERT prend ~20 s).
- Avoir 2 CV fictifs prêts (ex. `cv_01.pdf` Data Analyst, `cv_08.pdf` DevOps).
- Mettre le cache Gemini à jour la veille (appels déjà en cache → démo instantanée même sans quota).
- **Plan B** : captures d'écran / vidéo de la démo dans les slides ; plan C : notebook Colab.

### Questions probables du jury et réponses

**Pourquoi TF-IDF comme baseline ?**
Simple, rapide, sans entraînement, interprétable. Si SBERT ne fait pas mieux que TF-IDF, il ne se justifie pas :
c'est le rôle d'une baseline.

**Pourquoi Sentence-BERT et pas un modèle entraîné par vous ?**
Nous n'avons pas de paires (CV, offre) étiquetées en quantité suffisante pour entraîner. Un modèle pré-entraîné
multilingue capture le sens (« développeur » ≈ « developer », « apprentissage automatique » ≈ « machine learning »)
sans données d'entraînement.

**Pourquoi la similarité cosinus ?**
Elle compare l'orientation des vecteurs, pas leur longueur : un CV long et une offre courte peuvent être proches.
Les vecteurs sont normalisés, donc cosinus = produit scalaire (rapide sur 15 000 offres).

**Le score 0,72 veut dire 72 % de chances d'être embauché ?**
Non. C'est une proximité textuelle/sémantique. Il n'y a ni diplôme, ni expérience, ni salaire pris en compte.

**Comment avez-vous construit la vérité terrain ? Comment savez-vous qu'elle est fiable ?**
Pool par CV (top-6 TF-IDF ∪ top-6 SBERT ∪ 4 aléatoires), mélangé ; deux annotateurs indépendants avec une grille
écrite ; pertinent = les deux ont dit oui ; accord mesuré par le **kappa de Cohen** (donnez votre valeur :
> 0,6 = accord substantiel). Labels figés avant les tests finaux.

**Qu'est-ce que le biais de pooling ?**
Les offres annotées viennent des modèles évalués : une bonne offre qu'aucun des deux ne trouve n'est jamais
jugée. Les 4 offres aléatoires limitent ce biais sans l'éliminer.

**Qu'est-ce qu'une ablation ?**
On retire un composant et on mesure la perte : `delta = complet − ablaté`. Exemple : F1 (SBERT + nettoyage)
vs A1 (SBERT sans nettoyage) mesure l'apport du nettoyage.

**Gemini peut inventer des compétences (hallucination). Que faites-vous ?**
Température 0, prompt qui l'interdit, réponse JSON, puis un **garde-fou** : chaque compétence et chaque preuve
est recherchée dans le CV et dans l'offre ; ce qui n'y est pas est supprimé et compté (`guard_removed`).
C'est notre mesure de fidélité (L1 vs A2).

**Et si Gemini ne répond pas (quota, panne) ?**
Retry avec attente exponentielle, puis repli automatique sur l'explication à règles ; l'utilisateur voit un
avertissement. Le Top 3 ne dépend jamais de Gemini.

**Vie privée ?**
E-mail, téléphone, liens et adresse masqués avant tout log et tout envoi à Gemini ; masquage aussi à l'import
des 5054 CV ; aucun vrai CV dans Git.

**Pourquoi ne pas annoter les 5054 CV ?**
≈ 80 000 paires : irréaliste. 60 CV stratifiés pour la qualité, 5054 pour le passage à l'échelle et la cohérence.

**Limites ?**
Offres 2019-2022 ; pas de filtre diplôme/expérience/ville ; dictionnaire de compétences limité ; SBERT tronque
les longues offres ; petit échantillon annoté ; CV anglais vs offres françaises.

**Perspectives ?**
Re-ranking par cross-encoder sur le Top 20 ; filtres (ville, niveau d'études) ; extraction de compétences par NER
ou par LLM ; fine-tuning de SBERT avec les labels collectés ; offres récentes (scraping autorisé ou API).

**Qui a fait quoi ?**
Préparez une réponse claire par membre (Membre 1 : données + ranking + ablation ML ; Membre 2 : Gemini + API +
UI + ablation LLM, d'après la répartition M1.x / M2.x du code). Chacun doit pouvoir expliquer **tout** le pipeline.

### Conseils d'attitude

- Répondez par le **protocole** et les **chiffres mesurés**, jamais par « ça marche bien ».
- Si un delta est nul ou négatif, dites-le : « sur notre test, ce composant ne montre pas de bénéfice ». Le jury
  valorise l'honnêteté bien plus qu'un résultat flatteur.
- Si vous ne savez pas : « Je ne l'ai pas mesuré ; voici comment je le mesurerais : … ».
