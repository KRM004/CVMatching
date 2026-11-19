"""Interface Streamlit : upload d'un CV PDF -> Top 3 offres + scores + explications."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CV Matching", page_icon="🎯", layout="wide")
st.title("🎯 CV Matching & recommandation d'offres")
st.caption(
    "Le score mesure une proximité textuelle/sémantique entre votre CV et l'offre. "
    "Ce n'est PAS une probabilité d'embauche."
)

with st.sidebar:
    st.header("Options")
    explainer = st.radio("Explications", ["gemini", "rules"], format_func=lambda x: {
        "gemini": "Gemini (LLM)", "rules": "À règles (sans LLM)"}[x])  # fmt: skip
    compare = st.checkbox("Comparer Gemini et règles (même offre)", value=False)
    top_k = st.slider("Nombre d'offres", 1, 5, 3)
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        (st.success if h["status"] == "ok" else st.error)(
            f"API {h['status']} · {h['n_jobs']:,} offres · ranker : {h['ranker']}")
        if not h["gemini_configured"]:
            st.warning("Clé Gemini absente : explications à règles utilisées.")
        if h.get("error"):
            st.code(h["error"])
    except requests.RequestException:
        st.error(f"API injoignable ({API_URL})")

file = st.file_uploader("Votre CV (PDF numérique, fictif ou anonymisé)", type=["pdf"])
st.info("Vos coordonnées (e-mail, téléphone, adresse) sont masquées avant tout appel à Gemini.")


def call_api(name: str) -> dict:
    r = requests.post(
        f"{API_URL}/match",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        data={"top_k": top_k, "explain": "true", "explainer": name},
        timeout=180,
    )
    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(f"{r.status_code} : {detail}")
    return r.json()


def show_explanation(e: dict | None, label: str = "") -> None:
    if not e:
        st.caption("Pas d'explication.")
        return
    if label:
        st.markdown(f"**{label}**")
    st.write(e["explanation"] or "—")
    c1, c2 = st.columns(2)
    c1.markdown("✅ **Compétences en commun** : " + (", ".join(e["matching_skills"]) or "—"))
    c2.markdown("📈 **À développer** : " + (", ".join(e["missing_skills"]) or "—"))
    if e["evidence"]:
        with st.expander("Preuves (extraits du CV / de l'offre)"):
            for ev in e["evidence"]:
                st.write("• " + ev)
    if e.get("unverified_skills"):
        st.caption("⚠ Éléments non retrouvés dans les sources et retirés : " + ", ".join(e["unverified_skills"]))
    if e["source"] == "rules_fallback":
        st.caption(f"ℹ Gemini indisponible → explication à règles ({e.get('error')})")


if st.button("Analyser", type="primary", disabled=file is None):
    try:
        with st.spinner("Analyse en cours…"):
            res = call_api(explainer)
            res_cmp = call_api("rules" if explainer == "gemini" else "gemini") if compare else None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur : {exc}")
        st.stop()

    for w in res.get("warnings", []):
        st.warning(w)
    st.subheader("Compétences détectées dans votre CV")
    st.write(", ".join(res["resume_skills"]) or "Aucune compétence reconnue.")
    t = res["timings_ms"]
    st.caption(f"Ranker : {res['ranker']} · total {t.get('total', 0):.0f} ms "
               f"(classement {t.get('ranking', 0):.0f} ms, explications {t.get('explain', 0):.0f} ms)")  # fmt: skip

    st.subheader(f"Top {len(res['recommendations'])} des offres")
    for i, rec in enumerate(res["recommendations"], start=1):
        with st.container(border=True):
            head, score = st.columns([4, 1])
            head.markdown(f"### {i}. {rec['job_title']}")
            meta = " · ".join(x for x in [rec.get("company"), rec.get("city"), rec.get("sector")] if x)
            head.caption(meta)
            score.metric("Similarité", f"{rec['similarity_score']:.2f}")
            if res_cmp:
                a, b = st.columns(2)
                with a:
                    show_explanation(rec["explanation"], f"Explication « {explainer} »")
                with b:
                    other = res_cmp["recommendations"][i - 1]
                    show_explanation(other["explanation"], f"Explication « {'rules' if explainer == 'gemini' else 'gemini'} »")  # fmt: skip
            else:
                show_explanation(rec["explanation"])
            with st.expander("Description de l'offre"):
                st.write(rec["job_description"])
