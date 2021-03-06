class Config:
    MAX_LEN = 100
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    EPOCHS = 5
    BASE_MODEL = 'roberta-base'
    TRAIN_PATH = 'data/ner_dataset.csv'
    MODEL_PATH = 'entity_model.pt'


CONFIG = Config()


parent_class_mapping = {
    'B-*': 'O',
    'B-ADDRESS': 'O',
    'B-DATETIME': 'O',
    'B-DATETIME-DATE': 'O',
    'B-DATETIME-DATERANGE': 'O',
    'B-DATETIME-DURATION': 'O',
    'B-DATETIME-SET': 'O',
    'B-DATETIME-TIME': 'O',
    'B-DATETIME-TIMERANGE': 'O',
    'B-EMAIL': 'O',
    'B-EVENT': 'O',
    'B-EVENT-CUL': 'O',
    'B-EVENT-GAMESHOW': 'O',
    'B-EVENT-NATURAL': 'O',
    'B-EVENT-SPORT': 'O',
    'B-LOCATION': 'B-LOCATION',
    'B-LOCATION-GEO': 'B-LOCATION',
    'B-LOCATION-GPE': 'B-LOCATION',
    'B-LOCATION-STRUC': 'B-LOCATION',
    'B-MISCELLANEOUS': 'B-MISCELLANEOUS',
    'B-ORGANIZATION': 'B-ORGANIZATION',
    'B-ORGANIZATION-MED': 'B-ORGANIZATION',
    'B-ORGANIZATION-SPORTS': 'B-ORGANIZATION',
    'B-ORGANIZATION-STOCK': 'B-ORGANIZATION',
    'B-PERSON': 'B-PERSON',
    'B-PERSONTYPE': 'O',
    'B-PHONENUMBER': 'O',
    'B-PRODUCT': 'B-PRODUCT',
    'B-PRODUCT-AWARD': 'B-PRODUCT',
    'B-PRODUCT-COM': 'B-PRODUCT',
    'B-PRODUCT-LEGAL': 'B-PRODUCT',
    'B-QUANTITY': 'O',
    'B-QUANTITY-AGE': 'O',
    'B-QUANTITY-CUR': 'O',
    'B-QUANTITY-DIM': 'O',
    'B-QUANTITY-NUM': 'O',
    'B-QUANTITY-ORD': 'O',
    'B-QUANTITY-PER': 'O',
    'B-QUANTITY-TEM': 'O',
    'B-SKILL': 'O',
    'B-URL':  'O',
    'I-*': 'O',
    'I-ADDRESS': 'O',
    'I-DATETIME': 'O',
    'I-DATETIME-DATE': 'O',
    'I-DATETIME-DATERANGE': 'O',
    'I-DATETIME-DURATION': 'O',
    'I-DATETIME-SET': 'O',
    'I-DATETIME-TIME': 'O',
    'I-DATETIME-TIMERANGE': 'O',
    'I-EMAIL': 'O',
    'I-EVENT': 'O',
    'I-EVENT-CUL': 'O',
    'I-EVENT-GAMESHOW': 'O',
    'I-EVENT-NATURAL': 'O',
    'I-EVENT-SPORT': 'O',
    'I-LOCATION': 'I-LOCATION',
    'I-LOCATION-GEO': 'I-LOCATION',
    'I-LOCATION-GPE': 'I-LOCATION',
    'I-LOCATION-STRUC': 'I-LOCATION',
    'I-MISCELLANEOUS': 'I-MISCELLANEOUS',
    'I-ORGANIZATION': 'I-ORGANIZATION',
    'I-ORGANIZATION-MED': 'I-ORGANIZATION',
    'I-ORGANIZATION-SPORTS': 'I-ORGANIZATION',
    'I-PERSON': 'I-PERSON',
    'I-PERSONTYPE': 'O',
    'I-PHONENUMBER': 'O',
    'I-PRODUCT': 'I-PRODUCT',
    'I-PRODUCT-AWARD': 'I-PRODUCT',
    'I-PRODUCT-COM': 'I-PRODUCT',
    'I-PRODUCT-LEGAL': 'I-PRODUCT',
    'I-QUANTITY': 'O',
    'I-QUANTITY-AGE': 'O',
    'I-QUANTITY-CUR': 'O',
    'I-QUANTITY-DIM': 'O',
    'I-QUANTITY-NUM': 'O',
    'I-QUANTITY-ORD': 'O',
    'I-QUANTITY-PER': 'O',
    'I-QUANTITY-TEM': 'O',
    'I-SKILL': 'O',
    'I-URL': 'O',
    'O': 'O'
}

