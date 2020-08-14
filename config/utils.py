import re
import unicodedata


def preproc_single_sentence(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', s))
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[\x00-\x2F\x3A-\x40\x5B-\x60\x7B-\x7F]+", r" ", s)
    s = s.strip()
    return s