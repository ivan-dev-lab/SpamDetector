import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from translate import translate_data
from preprocess import clean_text, lemmatize_data

# функция для тренировки модели из sklearn

def train_model (fdata: str, models: list, paths_to_save: list, limit: int=5000) -> int:
    if len(models) == len(paths_to_save):
        data = pd.read_csv(fdata).iloc[0:limit]
        data = translate_data(data)
        data['text'] = clean_text(data)
        data['text'] = lemmatize_data(data)

        y = data['label'].values
        
        X = data.drop(['label'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)
        
        vectorizer = TfidfVectorizer()
        x_train_tfidf = vectorizer.fit_transform(raw_documents=x_train['text'])
        x_test_tfidf = vectorizer.transform(raw_documents=x_test['text'])

        i = 0
        for model in models:
            model_training = model()
            model_training.fit(x_train_tfidf,y_train)

            y_pred_current_model = model_training.predict(x_test_tfidf)
            accuracy_current_model = accuracy_score(y_test,y_pred_current_model)

            with open(paths_to_save[i], mode='wb+') as model_file:
                pickle.dump(model_training, model_file)
            i += 1


        with open(f"models/TF-IDF_vectorizer.pkl", mode='wb+') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        return accuracy_current_model
    else:
        raise ValueError(f'Несоотвествие длины списка моделей для обучения и списка путей сохранения [{len(models)}] != [{len(paths_to_save)}]')

# заготовки для функции обработки одного сообщения
# data = pd.read_csv('data/combined_data.csv').iloc[5006:5007]
# data = translate_data(data)
# data['text'] = clean_text(data)
# data['text'] = lemmatize_data(data)

# X = data['text']
# Y = data['label'].values
# print(f'текст = {X.values}', end='\n\n')
# print(f'реальное значение = {Y}')

# with open('models/TF-IDF_vectorizer.pkl', mode='rb') as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

# X_tfidf = vectorizer.transform(raw_documents=X)


# with open('models\MultinomialNB.pkl', mode='rb') as model_file:
#     model = pickle.load(model_file)
#     y_pred = model.predict(X_tfidf)

#     print(f'предсказанное значение = {y_pred}')