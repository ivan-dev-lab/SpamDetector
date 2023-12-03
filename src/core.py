import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from translate import translate_data
from preprocess import clean_text, lemmatize_data


# функция для тренировки модели из sklearn
def train_model (data: pd.DataFrame, model, fpath_save: str, limit: int=5000) -> int:
    
    data = pd.read_csv('data/combined_data.csv').iloc[0:limit]
    data = translate_data(data)
    data['text'] = clean_text(data)
    data['text'] = lemmatize_data(data)

    y = data['label'].values
    
    X = data.drop(['label'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)

    vectorizer = TfidfVectorizer()

    x_train_tfidf = vectorizer.fit_transform(raw_documents=x_train['text'])
    x_test_tfidf = vectorizer.transform(raw_documents=x_test['text'])

    model_training = model()
    model_training.fit(x_train_tfidf,y_train)

    y_pred_current_model = model_training.predict(x_test_tfidf)
    accuracy_current_model = accuracy_score(y_pred_current_model,y_test)

    with open(fpath_save, mode='wb') as model_file:
        pickle.dump(model_training, model_file)

    return accuracy_current_model