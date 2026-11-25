"""Text preprocessing used as the vectorizer `preprocessor` (report §4.2-§4.3).

All functions are module-level (no lambdas) so fitted pipelines can be pickled
with joblib. No NLTK data downloads are needed: stop words come from
scikit-learn and the Porter stemmer is rule-based.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import TOKEN_PATTERN

STOP_WORDS = sorted(ENGLISH_STOP_WORDS)

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CURRENCY_RE = re.compile(r"[£$€]")
PHONE_RE = re.compile(r"\d{5,}")
NUM_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z ]+")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(TOKEN_PATTERN)

NORMALIZATION_TOKENS = ["urltoken", "emailtoken", "moneytoken", "phonetoken", "numtoken"]


def clean_text(s, normalize=True):
    """Lowercase, normalise URLs/e-mails/currency/numbers, keep only letters.

    Steps (report §4.3):
      1. lowercase
      2. (normalize=True) URL -> urltoken, e-mail -> emailtoken,
         £ $ € -> moneytoken, digit runs of >= 5 -> phonetoken,
         other digit runs -> numtoken
      3. remove every character except a-z and spaces
    Tokenisation (>= 2 letters) and stop-word removal are done by the vectorizer.
    """
    s = str(s).lower()
    if normalize:
        s = URL_RE.sub(" urltoken ", s)
        s = EMAIL_RE.sub(" emailtoken ", s)
        s = CURRENCY_RE.sub(" moneytoken ", s)
        s = PHONE_RE.sub(" phonetoken ", s)
        s = NUM_RE.sub(" numtoken ", s)
    s = NON_ALPHA_RE.sub(" ", s)
    return MULTISPACE_RE.sub(" ", s).strip()


def clean_text_no_norm(s):
    """Ablation (c): same cleaning without number/URL/currency normalisation."""
    return clean_text(s, normalize=False)


def lowercase_only(s):
    """Ablation (e): raw text with only lowercasing."""
    return str(s).lower()


def tokenize(s):
    """Regex tokeniser: tokens of >= 2 letters (same as the vectorizer token_pattern)."""
    return TOKEN_RE.findall(s)


def tokens_no_stop(s):
    """Clean, tokenise and drop stop words (used for EDA word counts)."""
    return [t for t in tokenize(clean_text(s)) if t not in ENGLISH_STOP_WORDS]


_STEMMER = None


def stem_tokenize(s):
    """Ablation (d): tokenise, remove stop words, then Porter-stem each token.

    Stop words are removed *before* stemming so the stop list still matches.
    Input is expected to be already cleaned by `clean_text`.
    """
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer
        _STEMMER = PorterStemmer()
    return [_STEMMER.stem(t) for t in tokenize(s) if t not in ENGLISH_STOP_WORDS]


def _self_check():
    """Unit checks for clean_text; run with `python -m src.preprocess`."""
    cases = {
        "WINNER!! Claim £1000 now: call 09061234567": "winner claim moneytoken numtoken now call phonetoken",
        "Visit http://bit.ly/abc or www.win.com": "visit urltoken or urltoken",
        "Mail me at John.Doe@example.com ok?": "mail me at emailtoken ok",
        "Meet at 5, room 12": "meet at numtoken room numtoken",
        "I'm   gonna be HOME soon": "i m gonna be home soon",
    }
    for raw, expected in cases.items():
        got = clean_text(raw)
        assert got == expected, f"clean_text({raw!r}) -> {got!r}, expected {expected!r}"
    assert clean_text("Call 12345", normalize=False) == "call"
    assert tokenize("i m gonna be home") == ["gonna", "be", "home"]
    assert tokens_no_stop("I'm gonna be HOME soon") == ["gonna", "home", "soon"]
    assert stem_tokenize("winning prizes claimed") == ["win", "prize", "claim"]
    print("preprocess self-check: all assertions passed")
    for raw in cases:
        print(f"  {raw!r:50} -> {clean_text(raw)!r}")


if __name__ == "__main__":
    _self_check()
