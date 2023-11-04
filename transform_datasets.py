import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from sklearn.feature_extraction.text import CountVectorizer

def create_items_feature(genres, authors, year):
    try:
        concat = genres + ',' + authors + ',' + str(year)
    except:
        print(genres, authors, year)
    return concat


def create_users_feature(age, sex):
    concat = age + ',' + sex
    return concat

def custom_tokenizer(text):
    text = text.strip().split(',')
    text = [word.strip() for word in text]
    return text

def get_clean_datasets(path='data/'):

    interactions_df = pd.read_csv(path+'interactions_clean.csv')
    items_df = pd.read_csv(path+'items_clean.csv')
    users_df = pd.read_csv(path+'users_clean.csv')

    interactions_df = interactions_df[interactions_df['user_id'].isin(users_df['user_id'].unique())]
    interactions_df = interactions_df[interactions_df['item_id'].isin(items_df['id'].unique())]

    items_df['features'] = items_df.iloc[:, 2:].apply(lambda row: create_items_feature(*row), axis=1)
    users_df['features'] = users_df.iloc[:, 1:].apply(lambda row: create_users_feature(*row), axis=1)

    return interactions_df, items_df, users_df


def create_lightfm_dataset(path='data/'):
    
    interactions_df, items_df, users_df = get_clean_datasets(path=path)

    vectorizer_users = CountVectorizer(max_features=20000, lowercase=False, tokenizer=custom_tokenizer)
    vectorizer_items = CountVectorizer(max_features=50000, lowercase=False, tokenizer=custom_tokenizer)

    users_features = vectorizer_users.fit_transform(users_df['features'])
    items_features = vectorizer_items.fit_transform(items_df['features'])

    users_features_names = vectorizer_users.get_feature_names_out()
    items_features_names = vectorizer_items.get_feature_names_out()

    dataset = Dataset()

    user_ids_buffered = [x for x in interactions_df['user_id'].unique()]
    item_ids_buffered = [x for x in interactions_df['item_id'].unique()]

    user_ids_buffered = np.array(user_ids_buffered)
    item_ids_buffered = np.array(item_ids_buffered)

    dataset.fit(
        users=user_ids_buffered,
        items=item_ids_buffered,
        item_features=items_features_names,
        user_features=users_features_names
        )

    (interactions, weights) = dataset.build_interactions(((k, v, w)
                                                        for k, v, w in
                                                        zip(list(interactions_df['user_id'].values),
                                                            list(interactions_df['item_id'].values),
                                                            list(interactions_df['rating'].values+1))))

    return interactions_df, items_df, users_df, dataset, users_features, items_features, user_ids_buffered, item_ids_buffered