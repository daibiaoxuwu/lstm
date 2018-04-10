import spacy
nlp = spacy.load('en')
test_doc = nlp(u"The quick brown fox jump over the lazy dog.")
for token in test_doc:
    print(token, token.pos_, token.pos)
