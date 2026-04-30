import re
import spacy

nlp = spacy.blank("en")

COMPOUND_RE = re.compile(
    r'''
    \d+-[A-Za-z]+
    | [A-Za-z]+(?:-[A-Za-z]+)+
    | \$\d+(?:,\d{3})*(?:\.\d+)?
    | \d+(?:\.\d+)?%
    ''',
    re.VERBOSE,
)

# First regex: any number 1 or more followed by a hyphen followed by any alphabetical 1 or more
# Second regex: any alphabetical 1 or more followed by non-capturing group of hyphen followed by any 1 or more alphabeticals, 1 or more of that non-capturing group
# Third regex: dollar sign followed by 1 or more numbers, followed by non-cap group of comma and 3 numbers 0 or more, followed by an optional decimal point and 1 or more numbers.
# Fourth regex: number 1 or more followed by non-cap group of decimal point followed by 1 or more numbers (optional), then percent sign.


def tokenize(text):
    doc = nlp(text)

    tokens = [
        t.text.lower()
        for t in doc
        if not t.is_space and not t.is_punct and t.text.lower() != "'s"
    ]

    compounds = [x.lower() for x in COMPOUND_RE.findall(text)]

    return tokens + compounds