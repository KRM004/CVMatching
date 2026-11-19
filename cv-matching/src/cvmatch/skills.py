"""Extraction de compétences par dictionnaire (FR/EN) + extraits de preuve."""
from __future__ import annotations

import re
from functools import lru_cache

from .cleaning import fold

# canonique: alias1 | alias2 | ...   (comparaison sans accents ni casse)
_RAW = """
python: python
java: java
javascript: javascript | js
typescript: typescript
php: php
c++: c++
c#: c#
sql: sql | t-sql | pl/sql
r: langage r | r studio | rstudio
scala: scala
matlab: matlab
pandas: pandas
numpy: numpy
scikit-learn: scikit-learn | sklearn | scikit learn
tensorflow: tensorflow
pytorch: pytorch
keras: keras
machine learning: machine learning | apprentissage automatique | ml
deep learning: deep learning | apprentissage profond
nlp: nlp | traitement du langage naturel | natural language processing
computer vision: computer vision | vision par ordinateur
data analysis: data analysis | analyse de donnees | analyse des donnees | data analyst
data visualization: data visualization | visualisation de donnees | dataviz
statistiques: statistiques | statistics | statistique
power bi: power bi | powerbi
tableau: tableau software | tableau desktop
excel: excel | ms excel
spark: apache spark | pyspark | spark
hadoop: hadoop
airflow: airflow
etl: etl | talend | ssis
data warehouse: data warehouse | entrepot de donnees | datawarehouse
big data: big data
react: react | reactjs | react.js
angular: angular
vue.js: vue.js | vuejs
node.js: node.js | nodejs
django: django
flask: flask
fastapi: fastapi
spring boot: spring boot | springboot | spring framework
laravel: laravel
symfony: symfony
.net: .net | asp.net | dotnet
html/css: html | css | html5 | css3
rest api: rest | api rest | restful | api
graphql: graphql
docker: docker
kubernetes: kubernetes | k8s
git: git | github | gitlab
ci/cd: ci/cd | cicd | jenkins | gitlab ci | github actions
linux: linux | unix
aws: aws | amazon web services
azure: azure
gcp: gcp | google cloud
terraform: terraform
ansible: ansible
mysql: mysql
postgresql: postgresql | postgres
mongodb: mongodb
oracle: oracle
sql server: sql server | sqlserver
redis: redis
réseaux: reseaux | reseau | cisco | tcp/ip | networking
cybersécurité: cybersecurite | securite informatique | cybersecurity
scrum: scrum
agile: agile | methodes agiles
gestion de projet: gestion de projet | project management | chef de projet
comptabilité: comptabilite | comptable | accounting
sage: sage
audit: audit
contrôle de gestion: controle de gestion | reporting financier | budget
paie: paie | gestion de la paie
fiscalité: fiscalite | tva | impots
sap: sap | erp
recrutement: recrutement | sourcing | recruitment
gestion rh: ressources humaines | gestion des rh | sirh
droit du travail: droit du travail | code du travail
marketing digital: marketing digital | digital marketing
seo: seo | referencement
google ads: google ads | adwords | sea
community management: community management | reseaux sociaux | social media
négociation: negociation | negociations
prospection: prospection | business development | developpement commercial
relation client: relation client | service client | gestion de la relation client | crm
supply chain: supply chain | chaine logistique
logistique: logistique | logistics | gestion de stock | stocks
achats: achats | approvisionnement | procurement
qualité: qualite | assurance qualite | qhse
iso 9001: iso 9001 | iso9001
lean: lean | six sigma | amelioration continue
maintenance: maintenance | gmao
automatisme: automatisme | plc | siemens | automate programmable
autocad: autocad
revit/bim: revit | bim
génie civil: genie civil | batiment | btp
droit des affaires: droit des affaires | droit commercial | contrats
communication: communication | communication ecrite | communication orale
leadership: leadership | encadrement | management d'equipe | management
travail en équipe: travail en equipe | esprit d'equipe | teamwork
autonomie: autonomie | autonome
anglais: anglais | english
français: francais | french
arabe: arabe | arabic
"""


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[str, re.Pattern]]:
    out: list[tuple[str, re.Pattern]] = []
    for line in _RAW.strip().splitlines():
        canon, aliases = line.split(":", 1)
        alts = sorted({fold(a.strip()) for a in aliases.split("|") if a.strip()}, key=len, reverse=True)
        # pas de lettre/chiffre/+/# collé avant ou après (java != javascript, c != c++)
        pat = r"(?<![\w+#.])(?:" + "|".join(re.escape(a) for a in alts) + r")(?![\w+#])"
        out.append((canon.strip(), re.compile(pat)))
    return out


def extract_skills(text: str) -> list[str]:
    """Retourne les compétences détectées, dans l'ordre de première apparition."""
    if not text:
        return []
    t = fold(text)
    found: list[tuple[int, str]] = []
    for canon, pat in _patterns():
        m = pat.search(t)
        if m:
            found.append((m.start(), canon))
    return [c for _, c in sorted(found)]


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def find_evidence(text: str, skill: str, max_len: int = 180) -> str:
    """Phrase du texte contenant la compétence (pour justifier une correspondance)."""
    pats = dict(_patterns())
    pat = pats.get(skill)
    if pat is None or not text:
        return ""
    for sent in _SENT.split(text):
        s = sent.strip()
        if s and pat.search(fold(s)):
            return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"
    return ""
