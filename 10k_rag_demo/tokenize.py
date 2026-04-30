import spacy

def tokenize(text):
    nlp = spacy.blank('en')
    return nlp(text)