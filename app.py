import streamlit as st
import numpy as np
import pandas as pd
from lightfm import LightFM
from transform_datasets import *
import pickle
from datetime import datetime



st.title("Book recommendation system")

project_description = """

Описание на русском языке	

МТС - один из крупнейших мобильных операторов в России и СНГ.
МТС Библиотека - приложение для чтения электронных книг, прессы и прослушивания аудиокниг, доступно для абонентов всех мобильных операторов, продукт экосистемы МТС.

В представленном датасете собраны данные по пользователям и книгам, а также по их взаимодействиям (прочтение книги пользователем) из сервиса МТС Библиотека. Данные по чтению пользователями книг собраны за 2 два года, с 01-01-2018 по 31-12-2019 включительно, и разбавлены случайным шумом. ID пользователей и книг анонимизированы.


English Description

MTS is one of the largest mobile network operator in Russia and CIS.
MTS Library is the part of MTS business ecosystem. This service provides ebooks, audiobooks, and press.

The dataset contains user and book information along with their interactions. User reading statistics were collected for over 2 years (from 01-01-2018 till 31-12-2019) with random noise added. User IDs are anonymized.
"""

@st.cache_data
def get_all_data():
    path = 'data/'

    interactions_df, items_df, users_df, dataset, users_features, items_features, user_ids_buffered, item_ids_buffered = create_lightfm_dataset(path)
    st.write("Data has loaded")

    with open(path+'lightfm_model.pickle', 'rb') as handle:
        model = pickle.load(handle)
    st.write('Model has loaded')

    return interactions_df, items_df, users_df, dataset, users_features, items_features, user_ids_buffered, item_ids_buffered, model


interactions_df, items_df, users_df, dataset, users_features, items_features, user_ids_buffered, item_ids_buffered, model = get_all_data()

interactions_df['start_date'] = pd.to_datetime(interactions_df['start_date'])


def project_description_tab():
    st.header("Recommendation system interactive app")
    st.markdown("## Проект студентов группы МПИ-22-1 Александра Петрова и Рашида Мейланова")
    st.markdown("### Ссылка на датасет https://www.kaggle.com/datasets/sharthz23/mts-library/data")
    st.markdown("### Ссылка на гитхаб https://github.com/foolosopherr/book-recsys")
    st.write(project_description)

def slider_tab():
    st.header("The most popular books for the periods")
    st.subheader("Select a time interval")

    slider_value_1 = datetime(2018, 1, 1)
    slider_value_2 = datetime(2019, 12, 31)

    date_range = st.slider(
        "Select a time interval",
        value=(slider_value_1, slider_value_2))

    date_val_1, date_val_2 = date_range
    date_val_1, date_val_2 = pd.to_datetime(date_val_1), pd.to_datetime(date_val_2)

    if st.button('Compute results', key='tab2'):

        temp_books = interactions_df[(interactions_df['start_date'] >= date_val_1) & (interactions_df['start_date'] <= date_val_2)]
        temp_books = temp_books[['item_id', 'rating']].dropna().groupby('item_id')['rating'].agg(['mean', 'count']).sort_values(by=['mean', 'count'], ascending=False).index[:10]

        popular_books_ids = items_df[items_df['id'].isin(temp_books)]

        st.write()
        st.write()
        st.write()
        st.markdown('# The 10 most popular books in this period')
        for ind in popular_books_ids.index:
            st.markdown(f"### {popular_books_ids.loc[ind, 'title']}")
            st.write(f"             Genres: {popular_books_ids.loc[ind, 'genres'].replace(',', ', ')}")
            st.write(f"             Authors: {popular_books_ids.loc[ind, 'authors']}")
            st.write(f"             Year: {popular_books_ids.loc[ind, 'year']}")


def recommendation_tab():
    st.header("Recommendation Tab")

    all_users_list = users_df['user_id'].astype(str).unique()

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        option_user_id = st.selectbox(
            "Choose user_id to get reccomendations",
            all_users_list,
            index=0,
            placeholder="Select id",
            )
    with col2:
        option_n_recs = st.selectbox(
            "Select number of recommendations for user",
            [str(i) for i in range(1, 21)],
            index=9,
            placeholder="Select number",
            )
    
    if st.button('Compute results', key='tab3'):
        option_user_id = int(option_user_id)
        option_n_recs = int(option_n_recs)

        scores = model.predict(dataset.mapping()[0][option_user_id],
                            np.arange(dataset.interactions_shape()[1]),
                            user_features=users_features,
                            item_features=items_features)

        recommended_inds = item_ids_buffered[np.argsort(-scores)][:option_n_recs]
        already_rated_inds = interactions_df[interactions_df['user_id'] == option_user_id]['item_id'].values

        col1_, col2_ = st.columns(2)

        with col1_:
            st.write()
            st.markdown('## Recommendations')
            temp_rec = items_df[items_df['id'].isin(recommended_inds)]

            for ind in temp_rec.index:
                st.markdown(f"#### {temp_rec.loc[ind, 'title']}")
                st.write(f"             Genres: {temp_rec.loc[ind, 'genres'].replace(',', ', ')}")
                st.write(f"             Authors: {temp_rec.loc[ind, 'authors']}")
                st.write(f"             Year: {temp_rec.loc[ind, 'year']}")

        with col2_:
            st.write()
            st.markdown('## Books read')
            temp_read_books = items_df[items_df['id'].isin(already_rated_inds)]

            for ind in temp_read_books.index:
                st.markdown(f"#### {temp_read_books.loc[ind, 'title']}")
                st.write(f"             Genres: {temp_read_books.loc[ind, 'genres'].replace(',', ', ')}")
                st.write(f"             Authors: {temp_read_books.loc[ind, 'authors']}")
                st.write(f"             Year: {temp_read_books.loc[ind, 'year']}")




tab1, tab2, tab3 = st.tabs(["Project Description", "The most popular books for the periods", "Recommendations"])

with tab1:
    project_description_tab()
with tab2:
    slider_tab()
with tab3:
    recommendation_tab()
