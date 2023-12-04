from deep_translator import GoogleTranslator
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    filename='logs/preprocess.log',
                    filemode='w',
                    format="%(asctime)s %(levelname)s %(message)s",
                    encoding="utf8")

# из-за того, что файл с данными для обучения на английском языке, а проект создан для распознавания спама в русских письмах, исходные данные необходимо перевести на русский язык

def translate_data (data: pd.DataFrame, path_to_save: str) -> pd.DataFrame:
    translated_dict = {'label': [], 'text': []}
    i = 1

    for label,doc in data.values:
        logging.log(logging.INFO, f'перевод документа {i}/{len(data.values)} || {round((i*100)/len(data.values), 2)}%')
        if len(doc) > 5000: doc = doc[0:4999]
        try:
            translation = GoogleTranslator(source='en',target='ru').translate(doc)
            translated_dict['label'].append(label)
            translated_dict['text'].append(translation)
        except : 
            logging.log(logging.WARNING, f'Ошибка при переводе документа {i}')

        i+=1

    translated_data = pd.DataFrame(data=translated_dict)
    translated_data.to_csv(path_to_save)
    return translated_data