import pandas as pd
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re
import logging

logging.basicConfig(level=logging.INFO,
                    filename='logs/preprocess.log',
                    filemode='w',
                    format="%(asctime)s %(levelname)s %(message)s",
                    encoding="utf8")

def clean_text (data: pd.DataFrame) -> pd.DataFrame:
    cleaned_text = []
    sw = stopwords.words('russian')
    i = 1

    for doc in data['text']:
        cleaned_doc = nltk.word_tokenize(re.sub(r'\s+', ' ', re.sub(r'\W',' ',doc)).lower())
        cleaned_doc = [word for word in cleaned_doc if word not in sw]
        cleaned_text.append(' '.join(cleaned_doc))

        logging.log(level=logging.INFO, msg=f'очищен текст {i}/{len(data["text"])} || {round((i*100)/len(data["text"]), 2)}%')
        i+=1
    return cleaned_text

def lemmatize_data (data: pd.DataFrame) -> pd.DataFrame:
    lemmatized_text = []
    morph_analyzer = pymorphy2.MorphAnalyzer(lang='ru')
    i = 1
    for doc in data['text']:
        tokenized_doc = nltk.word_tokenize(doc)
        lemmatized_words = []
        for word in tokenized_doc:
            lemmatized_word = morph_analyzer.parse(word)[0]
            pos = lemmatized_word.tag.POS
            
            if pos == 'VERB' or pos == 'NOUN': 
                lemmatized_words.append(lemmatized_word.normal_form)

        lemmatized_text.append(' '.join(lemmatized_words))
        logging.log(level=logging.INFO, msg=f'лемматизирован текст {i}/{len(data["text"])} || {round((i*100)/len(data["text"]),2)}%')
        i+=1

    
    return lemmatized_text