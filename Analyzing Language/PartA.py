import spacy
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import csv
from collections import Counter
from spacy import displacy
from pathlib import Path

dataset = pd.read_csv("SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskB.txt",sep='\t', header=0,index_col=0,quoting=csv.QUOTE_NONE, error_bad_lines=False)
documents = dataset['Tweet text']
nlp = spacy.load('en_core_web_sm')

'''
1.	Tokenization
'''
num_of_tokens = 0
num_of_words = 0
total_words_length = 0
word_frequencies = Counter()
for document in documents:
    doc = nlp(document)
    num_of_tokens += len(doc)
    for sentence in doc.sents:
        words = []
        for token in sentence:
            if not token.is_punct and not token.is_space and not token.like_url and not token.is_digit and '@' not in token.text  and '#' not in token.text and token.is_alpha:
                total_words_length += len(token.text)
                words.append(token.text)
        num_of_words += len(words)
        word_frequencies.update(words)
print("Number of tokens: " + str(num_of_tokens))
print("Number of types: " + str(len(word_frequencies.keys())))
print("Number of words: " + str(sum(word_frequencies.values())))
print("Average number of words per tweet: " + str(sum(word_frequencies.values()) / len(documents)))
print("Average word length: " + str(total_words_length / sum(word_frequencies.values())))

'''
2.	POS-Tagging
'''
pos_tag_frequencies = Counter()
for document in documents:
    doc = nlp(document)
    for sentence in doc.sents:
        pos_tag = []
        for token in sentence:
            pos_tag.append("Finegrained: " + token.tag_ + " " + 'Universal: ' + token.pos_)
        pos_tag_frequencies.update(pos_tag)
most_frequent_POS_tags = sorted(pos_tag_frequencies.items(),key = lambda x:x[1],reverse = True)
print()
print(most_frequent_POS_tags[:10])

pos_tag_frequencies_with_precent = dict()
for key,value in pos_tag_frequencies.items():
    pos_tag_frequencies_with_precent[key] = value/num_of_tokens*100
most_frequent_POS_tags_with_precent = sorted(pos_tag_frequencies_with_precent.items(),key = lambda x:x[1],reverse = True)
print()
print(most_frequent_POS_tags_with_precent[:10])

pos_tag_token_frequencies_with_tokens = Counter()
for document in documents:
    doc = nlp(document)
    for sentence in doc.sents:
        pos_tag_token = []
        for token in sentence:
            pos_tag_token.append("Finegrained: " + token.tag_ + " " + 'Universal: ' + token.pos_ + " " + "Token: " +  token.text)
        pos_tag_token_frequencies_with_tokens.update(pos_tag_token)
pos_tag_token_frequencies_with_tokens = sorted(pos_tag_token_frequencies_with_tokens.items(),key = lambda x:x[1],reverse = True)
def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i])
        s = s.replace(',','') +'\n'
        file.write(s)
    file.close()
text_save('pos_tag_token_frequencies_with_tokens.txt',pos_tag_token_frequencies_with_tokens)



'''
4.  Lemmatization 
'''
lemmas = dict()
def find_inflections():
    for document in documents:
        doc = nlp(document)
        for sentence in doc.sents:
            for token in sentence:
                current = token.lemma_
                # not an inflection
                if current == token.text:
                    continue
                if current not in lemmas.keys():
                    lemmas[current] = dict()
                    lemmas[current]['counts'] = 1
                    lemmas[current]['sentences'] = list()
                    lemmas[current]['sentences'].append(str(sentence))

                    lemmas[current]['inflections'] = list()
                    lemmas[current]['inflections'].append(token.text)
                else:
                    lemmas[current]['counts'] += 1
                    # found another inflection
                    if token.text not in lemmas[current]['inflections']:
                        lemmas[current]['sentences'].append(str(sentence))
                        lemmas[current]['inflections'].append(token.text)
                        return current, lemmas[current]['inflections'], lemmas[current]['sentences']
lemma, inflections, sentences = find_inflections()
print()
print("Lemma:", lemma)
print("Inflections:", '\n', ", ".join(inflections))
print("Sentences:", '\n', ",\n".join(sentences))
print()



'''
5.	Named Entity Recognition
'''
ent = list()
ent_type = set()
for document in documents:
    doc = nlp(document)
    for entity in doc.ents:
        ent.append(entity.text)
        ent_type.add(entity.label_)
print("Number of named entities: " + str(len(ent)))
print("Number of different entity labels: " + str(len(ent_type)))


tweet1 = nlp('Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion  http://t.co/fej2v3OUBR')
tweet2 = nlp("@mrdahl87 We are rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)")
tweet3 = nlp("Hey there! Nice to see you Minnesota/ND Winter Weather")
tweets = [tweet1,tweet2,tweet3]
displacy.serve(tweets,style='ent')