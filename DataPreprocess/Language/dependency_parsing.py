
import spacy

nlp = spacy.load("en_core_web_sm")

while (True):
    input_text = input("Input a sentence: ")
    if(input_text == ""):
        break 

    doc = nlp(input_text)

    # Meaning of the token - https://spacy.io/api/token#attributes
    # **POS listing**
    # ADJ: adjective
    # ADP: adposition
    # ADV: adverb
    # AUX: auxiliary
    # CCONJ: coordinating conjunction
    # DET: determiner
    # INTJ: interjection
    # NOUN: noun
    # NUM: numeral
    # PART: particle
    # PRON: pronoun
    # PROPN: proper noun
    # PUNCT: punctuation
    # SCONJ: subordinating conjunction
    # SYM: symbol
    # VERB: verb

    # text = original text , lemma_ = lemmatized base form , pos_ = Parts of speech
    # head.text = the word on which this current token is dependent, head.pos_ = Parts of speech of the head
    # token.children = words that are modifying this token 

    for token in doc:
        # print(token.text, token.lemma_, token.pos_, token.head.text, token.head.pos_,[child for child in token.children])
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            print("Subject = ", token.text)
            for child in token.children:
                if child.pos_ == "ADJ":
                    print("modifier = ", child)
