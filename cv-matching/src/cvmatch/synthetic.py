"""Données FICTIVES (smoke tests, démo hors-ligne). À ne jamais utiliser pour rapporter des scores.

- make_synthetic_jobs() : offres fictives au schéma Rekrute (mêmes colonnes que le vrai dataset)
- DEMO_FAMILIES / make_demo_cvs() : 20 CV fictifs (un par famille de métiers) avec e-mail,
  téléphone et adresse factices pour tester le masquage PII.
"""
from __future__ import annotations

import random

import pandas as pd

# (id, intitulés, compétences, mission, formation, secteur, fonction, mots-clés de pertinence "proxy")
DEMO_FAMILIES = [
    ("data_analyst", ["Data Analyst", "Analyste de données", "Business Data Analyst"],
     ["Python", "SQL", "Power BI", "Excel", "Pandas", "statistiques"],
     "Collecter, nettoyer et analyser les données pour produire des tableaux de bord",
     "Bac +5 en informatique, statistiques ou data", "Informatique / Télécoms", "Etudes et statistiques",
     ["data analyst", "analyste de donnees", "business intelligence"]),
    ("data_scientist", ["Data Scientist", "Ingénieur Data Science", "Machine Learning Scientist"],
     ["Python", "machine learning", "scikit-learn", "Pandas", "SQL", "statistiques"],
     "Concevoir, entraîner et évaluer des modèles prédictifs sur des données volumineuses",
     "Bac +5 en data science ou mathématiques appliquées", "Informatique / Télécoms", "Etudes et statistiques",
     ["data scien", "machine learning", "data science"]),
    ("ml_engineer", ["Ingénieur Machine Learning", "NLP Engineer", "Ingénieur IA"],
     ["Python", "PyTorch", "NLP", "Docker", "deep learning", "Git"],
     "Industrialiser des modèles de deep learning et de traitement du langage naturel",
     "Bac +5 ingénieur en informatique ou IA", "Informatique / Télécoms", "Recherche et développement",
     ["machine learning", "nlp", "ingenieur ia", "intelligence artificielle", "deep learning"]),
    ("java_dev", ["Développeur Java", "Ingénieur Développement Java/Spring", "Développeur Back-end Java"],
     ["Java", "Spring Boot", "SQL", "Git", "REST API", "Docker"],
     "Développer et maintenir des services back-end et des API REST",
     "Bac +5 ingénieur ou master informatique", "Informatique / Télécoms", "Développement informatique",
     ["java"]),
    ("fullstack", ["Développeur Full Stack", "Développeur JavaScript", "Ingénieur Full Stack"],
     ["JavaScript", "React", "Node.js", "MongoDB", "REST API", "Git"],
     "Concevoir des applications web complètes, du front-end au back-end",
     "Bac +3 à Bac +5 en informatique", "Informatique / Télécoms", "Développement informatique",
     ["full stack", "fullstack", "javascript"]),
    ("frontend", ["Développeur Front-end", "Intégrateur Web React", "Développeur Angular"],
     ["JavaScript", "Angular", "HTML", "CSS", "TypeScript", "Git"],
     "Intégrer des maquettes et développer des interfaces web réactives",
     "Bac +3 à Bac +5 en informatique", "Informatique / Télécoms", "Développement informatique",
     ["front", "angular", "react", "integrateur web"]),
    ("php_dev", ["Développeur PHP Laravel", "Développeur Web PHP", "Ingénieur Symfony"],
     ["PHP", "Laravel", "MySQL", "HTML", "CSS", "Git"],
     "Développer des sites et applications web sur mesure en PHP",
     "Bac +3 à Bac +5 en informatique", "Informatique / Télécoms", "Développement informatique",
     ["php", "laravel", "symfony"]),
    ("devops", ["Ingénieur DevOps", "Cloud Engineer", "DevOps Engineer"],
     ["Docker", "Kubernetes", "Jenkins", "AWS", "Linux", "Terraform", "Ansible"],
     "Automatiser les déploiements et administrer l'infrastructure cloud",
     "Bac +5 ingénieur en informatique ou réseaux", "Informatique / Télécoms", "Systèmes et réseaux",
     ["devops", "cloud", "sre"]),
    ("sysadmin", ["Administrateur Systèmes et Réseaux", "Ingénieur Réseaux", "Technicien Support Réseaux"],
     ["Linux", "réseaux", "Cisco", "cybersécurité", "Azure"],
     "Administrer les serveurs, les réseaux et assurer la sécurité du système d'information",
     "Bac +3 à Bac +5 en réseaux et télécoms", "Informatique / Télécoms", "Systèmes et réseaux",
     ["reseau", "systeme", "administrateur"]),
    ("it_pm", ["Chef de projet IT", "Product Owner", "Scrum Master"],
     ["gestion de projet", "Scrum", "Agile", "communication", "leadership", "anglais"],
     "Piloter des projets informatiques et animer les équipes en méthodes agiles",
     "Bac +5 ingénieur ou école de commerce", "Informatique / Télécoms", "Gestion de projet",
     ["chef de projet", "product owner", "scrum"]),
    ("accountant", ["Comptable", "Aide-comptable", "Chef comptable"],
     ["comptabilité", "Sage", "fiscalité", "TVA", "Excel", "paie"],
     "Tenir la comptabilité générale, préparer les déclarations de TVA et la paie",
     "Bac +3 à Bac +5 en comptabilité ou finance", "Banque / Finance", "Comptabilité",
     ["comptable", "comptabilite"]),
    ("controller", ["Contrôleur de gestion", "Analyste financier", "Contrôleur de gestion junior"],
     ["contrôle de gestion", "Excel", "SAP", "reporting financier", "budget"],
     "Élaborer les budgets, suivre les écarts et produire le reporting financier",
     "Bac +5 en finance ou contrôle de gestion", "Banque / Finance", "Finance",
     ["controleur de gestion", "controle de gestion", "analyste financier"]),
    ("hr", ["Chargé de recrutement", "Responsable RH", "Gestionnaire RH"],
     ["recrutement", "sourcing", "droit du travail", "paie", "SIRH", "communication"],
     "Gérer le recrutement, l'administration du personnel et la paie",
     "Bac +3 à Bac +5 en ressources humaines", "Services", "Ressources humaines",
     ["rh", "ressources humaines", "recrutement"]),
    ("marketing", ["Responsable Marketing Digital", "Chargé de communication digitale", "Community Manager"],
     ["marketing digital", "SEO", "Google Ads", "community management", "communication"],
     "Définir et piloter la stratégie digitale et animer les réseaux sociaux",
     "Bac +3 à Bac +5 en marketing ou communication", "Marketing / Communication", "Marketing",
     ["marketing", "community manager", "communication digitale", "digital"]),
    ("sales", ["Commercial B2B", "Business Developer", "Attaché commercial"],
     ["négociation", "prospection", "CRM", "relation client", "communication", "anglais"],
     "Prospecter, négocier avec les clients et développer le portefeuille",
     "Bac +3 à Bac +5 en commerce", "Commerce / Vente", "Commercial",
     ["commercial", "business developer", "vente", "ventes"]),
    ("supply", ["Responsable Supply Chain", "Logisticien", "Gestionnaire de stock"],
     ["supply chain", "logistique", "achats", "SAP", "Excel", "gestion de stock"],
     "Optimiser les flux, les approvisionnements et la gestion des stocks",
     "Bac +3 à Bac +5 en logistique ou supply chain", "Industrie", "Logistique",
     ["supply chain", "logisti", "stock", "approvisionnement"]),
    ("quality", ["Ingénieur Qualité", "Responsable QHSE", "Technicien Qualité"],
     ["qualité", "ISO 9001", "lean", "audit", "amélioration continue"],
     "Déployer le système qualité, conduire les audits et l'amélioration continue",
     "Bac +5 ingénieur qualité ou industriel", "Industrie", "Qualité",
     ["qualite", "qhse"]),
    ("maintenance", ["Ingénieur Maintenance", "Technicien de Maintenance Industrielle", "Responsable GMAO"],
     ["maintenance", "GMAO", "automatisme", "Siemens", "lean"],
     "Assurer la maintenance préventive et corrective des équipements industriels",
     "Bac +2 à Bac +5 en électromécanique ou maintenance", "Industrie", "Maintenance",
     ["maintenance", "gmao"]),
    ("civil", ["Ingénieur Génie Civil", "Conducteur de Travaux", "Ingénieur Bâtiment"],
     ["génie civil", "AutoCAD", "Revit", "BTP", "gestion de projet"],
     "Suivre les chantiers, établir les plans et piloter les travaux",
     "Bac +5 ingénieur génie civil", "BTP / Construction", "Travaux",
     ["genie civil", "conducteur de travaux", "batiment", "travaux"]),
    ("legal", ["Juriste d'entreprise", "Responsable juridique", "Assistant juridique"],
     ["droit des affaires", "contrats", "droit du travail", "communication", "anglais"],
     "Rédiger et négocier les contrats, assurer la veille et le conseil juridique",
     "Bac +5 en droit des affaires", "Services", "Juridique",
     ["juriste", "juridique"]),
]

_CITIES = ["Casablanca", "Rabat", "Tanger", "Marrakech", "Fès", "Agadir", "Kénitra", "Salé"]
_COMPANIES = ["Société Alpha", "Groupe Atlas", "Maroc Digital SARL", "Delta Industries", "Orion Conseil",
              "Sahara Tech", "Nord Logistics", "Cabinet Horizon"]  # fmt: skip


def make_synthetic_jobs(n_per_family: int = 15, seed: int = 0, n_exact_duplicates: int = 5) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, next_id = [], 100000
    for fam in DEMO_FAMILIES:
        _, titles, skills, mission, edu, sector, function, _ = fam
        for _ in range(n_per_family):
            sk = rng.sample(skills, k=min(len(skills), rng.randint(4, len(skills))))
            title = rng.choice(titles) + rng.choice(["", " (H/F)", " - Confirmé", " Junior"])
            rows.append(
                {
                    "job_offer_id": next_id,
                    "job_title": title,
                    "city": rng.choice(_CITIES),
                    "function": function,
                    "sector": sector,
                    "Expérience requise": f"De {rng.randint(0, 5)} à {rng.randint(5, 8)} ans",
                    "Niveau d'étude et formation": edu,
                    "Entreprise": rng.choice(_COMPANIES),
                    "Poste": f"{mission}. Vous utiliserez au quotidien : {', '.join(sk[:4])}.",
                    "Profil recherché": f"{edu}. Maîtrise de {', '.join(sk)}. Bonne communication.",
                    "Qualités requises pour réussir dans ce rôle": "Rigueur, autonomie, esprit d'équipe.",
                    "soft_skills": "autonomie, travail en équipe",
                    "pub_date_year": rng.randint(2019, 2022),
                }
            )
            next_id += 1
    df = pd.DataFrame(rows)
    dups = df.sample(n=n_exact_duplicates, random_state=seed).copy()
    dups["job_offer_id"] = range(900000, 900000 + len(dups))
    return pd.concat([df, dups], ignore_index=True)


def make_demo_cvs() -> dict[str, dict]:
    """20 CV fictifs : {cv_id: {"text": ..., "target_keywords": [...], "family": ...}}."""
    cvs = {}
    for i, fam in enumerate(DEMO_FAMILIES, start=1):
        fid, titles, skills, mission, edu, _, _, kws = fam
        cv_id = f"cv_{i:02d}"
        text = f"""CANDIDAT {i:02d} (profil fictif de démonstration)
Email : candidat{i:02d}@example.com | Tél : +212 6 12 34 {50 + i} 78
Adresse : {i} rue Exemple, Rabat
LinkedIn : linkedin.com/in/candidat{i:02d}

{titles[0].upper()}

PROFIL
Professionnel motivé, {titles[1].lower()}, souhaitant évoluer dans le domaine. Rigueur et autonomie.

COMPÉTENCES
{', '.join(skills)}

EXPÉRIENCE
- 2022-2024 : {titles[1]} chez Société Fictive {i}. {mission}. Outils : {', '.join(skills[:3])}.
- 2020-2022 : Stage puis poste junior. Utilisation de {', '.join(skills[-2:])} sur des projets réels.

FORMATION
- {edu}

LANGUES : Français, Anglais, Arabe
"""
        cvs[cv_id] = {"text": text, "target_keywords": kws, "family": fid}
    return cvs
