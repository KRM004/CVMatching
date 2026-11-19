import re

from cvmatch.cleaning import TOKEN_PATTERN, clean_text, raw_text
from cvmatch.pii import mask_pii
from cvmatch.skills import extract_skills, find_evidence


def test_clean_preserves_technical_terms():
    t = clean_text("<p>Expert C++, C# et .NET ; Node.js &amp; SQL</p> https://x.com")
    for term in ["C++", "C#", ".NET", "Node.js", "SQL"]:
        assert term in t
    assert "<p>" not in t and "http" not in t and "&amp;" not in t


def test_tokenizer_keeps_tech_tokens():
    toks = re.findall(TOKEN_PATTERN, "c++ c# .net node.js asp.net python,")
    assert toks == ["c++", "c#", ".net", "node.js", "asp.net", "python"]


def test_raw_text_is_untouched():
    assert raw_text("<b>x</b>") == "<b>x</b>"
    assert raw_text(None) == ""


def test_pii_masking():
    cv = (
        "Jean Test\nEmail : jean.test@example.com | Tél : +212 6 12 34 56 78\n"
        "Adresse : 12 rue Exemple, Rabat\nlinkedin.com/in/jeantest\nPython et SQL"
    )
    m = mask_pii(cv)
    for secret in ["jean.test@example.com", "+212", "12 34 56 78", "rue Exemple", "jeantest"]:
        assert secret not in m
    assert "Python" in m and "[EMAIL]" in m


def test_skills_extraction_boundaries():
    sk = extract_skills("Python, SQL et Power BI. Java et JavaScript. C++ / C#")
    assert {"python", "sql", "power bi", "java", "javascript", "c++", "c#"} <= set(sk)
    assert "java" not in extract_skills("Seulement JavaScript")
    assert "comptabilité" in extract_skills("Comptabilité générale")  # accents ignorés


def test_find_evidence_returns_sentence():
    ev = find_evidence("Je maîtrise Docker. Aussi Git.", "docker")
    assert "Docker" in ev
