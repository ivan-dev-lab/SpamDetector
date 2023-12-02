from deep_translator import GoogleTranslator
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    filename='logs/translate.log',
                    filemode='w',
                    format="%(asctime)s %(levelname)s %(message)s",
                    encoding="utf8")

# из-за того, что файл с данными для обучения на английском языке, а проект создан для распознавания спама в русских письмах, исходные данные необходимо перевести на русский язык

def translate_data (data: pd.DataFrame) -> pd.DataFrame:
    translated_dict = {'label': [], 'text': []}
    i = 1

    for label,doc in data.values:
        logging.log(logging.INFO, f'{i}/{len(data.values)} || {(i*100)/len(data.values)}%')
        try:
            translation = GoogleTranslator(source='en',target='ru').translate(doc)
            translated_dict['label'].append(label)
            translated_dict['text'].append(translation)
        except: 
            logging.log(logging.WARNING, f'Ошибка при переводе документа {i}')
        i+=1

    translated_data = pd.DataFrame(data=translated_dict)
    return translated_data