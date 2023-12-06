import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from deep_translator import GoogleTranslator
import langdetect
import os
from translate import translate_data
from preprocess import clean_text, lemmatize_data

# функция для тренировки моделей из sklearn
def train_models (fdata: str, models: list, paths_to_save: list, limit=None, need_translate: bool=False) -> int:
    if len(models) == len(paths_to_save):
        if limit != None and limit > 0: data = pd.read_csv(fdata).iloc[0:limit]
        else: data = pd.read_csv(fdata)

        if need_translate: data = translate_data(data)
        data['text'] = clean_text(data['text'])
        data['text'] = lemmatize_data(data['text'])

        y = data['label'].values
        
        X = data.drop(['label'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)
        
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(raw_documents=X_train['text'])
        X_test_tfidf = vectorizer.transform(raw_documents=X_test['text'])

        i = 0
        for model in models:
            print(model, end=': ')
            model_training = model()
            model_training.fit(X_train_tfidf,y_train)

            y_pred_current_model = model_training.predict(X_test_tfidf)
            accuracy_current_model = accuracy_score(y_test,y_pred_current_model)
            print(accuracy_current_model)

            with open(paths_to_save[i], mode='wb+') as model_file:
                pickle.dump(model_training, model_file)
            i += 1


        with open(f"models/TF-IDF_vectorizer.pkl", mode='wb+') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        return accuracy_current_model
    else:
        raise ValueError(f'Несоотвествие длины списка моделей для обучения и списка путей сохранения [{len(models)}] != [{len(paths_to_save)}]')

def get_predictions (texts: list, fpath_models: str) -> list:
    if os.path.exists(fpath_models) == False:
        raise ValueError(f'Директории "{fpath_models}" не существует.')

    models = []
    for fpath in os.listdir(fpath_models):
        with open (f'{fpath_models}{fpath}', mode='rb') as file:
            if 'TF-IDF' in fpath: vectorizer = pickle.load(file)
            elif ('.zip' in fpath) == False: models.append(pickle.load(file))

    X_df = pd.DataFrame(data=texts, columns=['text'])
    for index, X_text in enumerate(X_df.values):
        if langdetect.detect(X_text[0]) != 'ru':
            translation = GoogleTranslator(target='ru').translate(X_text[0])
            X_df.iloc[index] = translation

    X_df['text'] = clean_text(X_df['text'])
    X_df['text'] = lemmatize_data(X_df['text'])
    print(X_df['text'])
    X_vectorized = vectorizer.transform(raw_documents=X_df['text'])
    
    predictions = []
    for doc in X_vectorized:
        current_predictions = []
        for model in models:
            y_pred = model.predict(doc)
            current_predictions.append(y_pred[0])
        predictions.append(current_predictions)
        
    
    return predictions