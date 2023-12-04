from deep_translator import GoogleTranslator
import pandas as pd
import logging
import asyncio
import httpx

logging.basicConfig(level=logging.INFO,
                    filename='logs/preprocess.log',
                    filemode='w',
                    format="%(asctime)s %(levelname)s %(message)s",
                    encoding="utf8")

# из-за того, что файл с данными для обучения на английском языке, а проект создан для распознавания спама в русских письмах, исходные данные необходимо перевести на русский язык

async def translate_doc (label: str, doc: str, index: int, data_length: int) -> pd.DataFrame:
    index+=1
    logging.log(logging.INFO, f'перевод документа {index}/{data_length} || {round((index*100)/data_length, 3)}%')
    if len(doc) > 5000: doc = doc[0:4999]
    try:
        translation = GoogleTranslator(source='auto',target='ru').translate(doc)
        return (label,translation)
    except Exception as error: 
        logging.log(logging.ERROR, f'Ошибка при переводе документа {index} - {error}')
        return (None, None)
    

async def translate_data (data: pd.DataFrame, path_to_save: str):
    translated_dict = {'label': [], 'text': []}

    tasks = [translate_doc(label, doc, index, len(data)) for index, label, doc in data.itertuples(index=True)]

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*tasks)

        for label, translation in results:
            if label != None and translation != None:
                translated_dict['label'].append(label)
                translated_dict['text'].append(translation)

    translated_data = pd.DataFrame(data=translated_dict)
    translated_data.to_csv(path_to_save)
    return translated_data