import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from deep_translator import GoogleTranslator
import langdetect
import os
import sys
from translate import translate_data
from preprocess import clean_text, lemmatize_data

# функция для тренировки моделей из sklearn
def train_models (fdata: str, models: list, paths_to_save: list, limit: int=5000) -> int:
    if len(models) == len(paths_to_save):
        data = pd.read_csv(fdata).iloc[0:limit]
        data = translate_data(data)
        data['text'] = clean_text(data['text'])
        data['text'] = lemmatize_data(data['text'])

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


def get_predictions (texts: list) -> list:
    all_predictions = []

    for text in texts:
        if len(text) > 5000: text = text[0:4999]
        
        if langdetect.detect(text) != 'ru':
            text = GoogleTranslator(source='en',target='ru').translate(text)

        text_df = pd.DataFrame(data={'text': text}, index=[0])

        text_df['text'] = clean_text(text_df['text'])
        text_df['text'] = lemmatize_data(text_df['text'])

        with open('models/TF-IDF_vectorizer.pkl', mode='rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        text_df_TFIDF = vectorizer.transform(text_df)
        current_predictions = []

        for path_model in os.listdir('models/'):
            with open(f'models/{path_model}', mode='rb') as model_file:
                if 'TF-IDF' in path_model: break
                model = pickle.load(model_file)

            current_predictions.append(model.predict(text_df_TFIDF)[0])
        
        all_predictions.append(current_predictions)

    print(all_predictions)

emails = [
    'Уральским федеральным университетом:Изумруд 2023/24, Информатика, 11 класс (не выполнено). Изумруд 2023/24, Математика, 11 класс (не выполнено). Изумруд 2023/24, Русский язык, 11 класс (не выполнено). Тест отборочного этапа необходимо выполнить до 14 января 2024 года 21:59 по московоскому времени (23:59 по времени города Екатеринбурга). Для доступа в систему тестирования используйте логин ivan_ul и пароль, указанный при регистрации. Если вы забыли пароль, то его можно сменить по ссылке: dovuz.urfu.ru/user/forgot. Обратиться в службу поддержки можно в чате вк.'
]
# модели хорошо понимают где спам, но плохо отличают письма не спама
get_predictions(emails)
