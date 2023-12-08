import streamlit as st
import numpy as np
import pandas as pd
from lightfm_functions import (
    get_clean_dataframes,
    get_recommendation,
    create_lightfm_dataset,
    train_lightfm_model,
)
from plot_functions import (
    create_full_interactions_df,
    plot_map,
    popular_books,
    show_book_cover,
)
import pickle
from PIL import Image
import requests
import random


st.title("Book recommendation system")

project_description = """

Описание

В течение последних нескольких десятилетий, с появлением Youtube, Amazon, Netflix и многих других подобных веб-сервисов, рекомендательные системы занимают все больше и больше места в нашей жизни. От электронной коммерции (предлагайте покупателям статьи, которые могут их заинтересовать) до онлайн-рекламы (предлагайте пользователям правильный контент, соответствующий их предпочтениям), рекомендательные системы сегодня неизбежны в наших ежедневных онлайн-путешествиях.
В самом общем смысле, рекомендательные системы — это алгоритмы, направленные на предложение пользователям соответствующих товаров (таких как фильмы для просмотра, текст для чтения, продукты для покупки или что-то еще, в зависимости от отрасли).

Рекомендательные системы действительно имеют решающее значение в некоторых отраслях, поскольку они могут приносить огромный доход, если они эффективны, а также позволяют значительно выделиться среди конкурентов. В качестве доказательства важности рекомендательных систем можно упомянуть, что несколько лет назад Netflix организовал конкурс («премия Netflix»), целью которого было создание рекомендательной системы, которая работает лучше, чем ее собственный алгоритм, с призом. 1 миллион долларов, чтобы выиграть.
"""


@st.cache_data
def get_all_data():
    path = "data/"
    with st.spinner("Getting dataset..."):
        interactions_df, items_df, users_df = get_clean_dataframes(path)
    with st.spinner("Getting model..."):
        with open(path + "lightfm_model.pickle", "rb") as handle:
            model = pickle.load(handle)

    with st.spinner("Creating LightFM dataset..."):
        (
            dataset,
            user_ids_buffered,
            item_ids_buffered,
            interactions,
            weights,
        ) = create_lightfm_dataset(interactions_df)

    full_interactions_df = create_full_interactions_df(
        interactions_df, users_df, items_df
    )
    return (
        interactions_df,
        full_interactions_df,
        items_df,
        users_df,
        dataset,
        user_ids_buffered,
        item_ids_buffered,
        interactions,
        weights,
        model,
    )


(
    interactions_df,
    full_interactions_df,
    items_df,
    users_df,
    dataset,
    user_ids_buffered,
    item_ids_buffered,
    interactions,
    weights,
    model,
) = get_all_data()


def project_description_tab():
    st.markdown(
        "## Проект студентов группы МПИ-22-1 Александра Петрова и Рашида Мейланова"
    )
    st.markdown(
        "#### Ссылка на датасет https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data"
    )
    st.markdown("#### Ссылка на гитхаб https://github.com/foolosopherr/book-recsys")
    st.write("")
    st.write("")
    st.write("")
    st.write(project_description)


def map_tab():
    st.header("Map of users")

    agg_fun_option = st.selectbox(
        "Choose aggregation function",
        [
            "Number of ratings by Country",
            "Number of users by Country",
            "Mean rating by Country",
        ],
        index=0,
    )

    if agg_fun_option == "Number of ratings by Country":
        fig = plot_map(full_interactions_df, "count")

    elif agg_fun_option == "Number of users by Country":
        fig = plot_map(
            full_interactions_df[["user_id", "rating", "Country Code"]].drop_duplicates(
                subset=["user_id", "Country Code"], keep="first"
            ),
            "count",
        )

    else:
        fig = plot_map(full_interactions_df, "mean")

    st.plotly_chart(fig)


def popular_books_tab():
    st.header("Most popular books")

    col1, col2, col3 = st.columns([0.35, 0.4, 0.25])
    with col1:
        age_filter = st.selectbox(
            "Choose age",
            [
                "Не использовать",
                "0-10",
                "11-19",
                "19-30",
                "31-40",
                "41-60",
                "61+",
                "Age_Unknown",
            ],
            index=0,
            placeholder="Choose age",
        )
    with col2:
        country_filter = st.selectbox(
            "Choose country",
            ["Не использовать"]
            + full_interactions_df["Country"].value_counts().index.values.tolist(),
            index=0,
            placeholder="Choose country",
        )
    with col3:
        n_top_filter = st.number_input(
            "Choose number of books",
            1,
            100,
            value=10,
            placeholder="Write n (min=1, max=100)",
        )

    if age_filter == "Не использовать":
        age_filter = None

    if country_filter == "Не использовать":
        country_filter = None

    if st.button("Show most popular books", key="tab_popular_books_tab"):
        popularBooks = popular_books(
            full_interactions_df, n_top_filter, age_filter, country_filter
        )
        if type(popularBooks) == str:
            st.markdown("## При таких фильтрах нет книг")

        else:
            for ind in popularBooks.index:
                col1_1, col1_2 = st.columns([0.6, 0.4])
                with col1_1:
                    st.markdown(f"#### {popularBooks.loc[ind, 'Title'].values[0]}")
                    st.write(f"             Author: {popularBooks.loc[ind, 'Author']}")
                    st.write(
                        f"             Publication Year: {popularBooks.loc[ind, 'Publication Year']}"
                    )
                    st.write(
                        f"             Publisher: {popularBooks.loc[ind, 'Publisher']}"
                    )
                    st.write(
                        f"             Popularity: {popularBooks.loc[ind, 'Popularity']}"
                    )
                    st.write(
                        f"             Average Rating: {popularBooks.loc[ind, 'AverageRatings']}"
                    )

                with col1_2:
                    st.image(
                        show_book_cover(popularBooks.loc[ind, "image_url"]),
                        use_column_width=True,
                    )


def recommendation_tab():
    st.header("Recommendation Tab")

    all_users_list = users_df["user_id"].astype(str).unique()

    col1, col2, col3 = st.columns(3, gap="medium")
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
    with col3:
        task = st.selectbox(
            "What to show",
            ["Recommendations", "Books the user read"],
        )

    if st.button("Compute results", key="tab_recommendations"):
        option_user_id = int(option_user_id)
        option_n_recs = int(option_n_recs)

        already_rated_inds, recommended_inds = get_recommendation(
            model,
            dataset,
            interactions_df,
            option_user_id,
            item_ids_buffered,
            option_n_recs,
        )

        if task == "Recommendations":
            temp_rec = items_df[items_df["item_id"].isin(recommended_inds)]
            st.markdown("## Recommendations")

        else:
            temp_rec = items_df[items_df["item_id"].isin(already_rated_inds)]
            temp_rec = pd.merge(
                items_df,
                interactions_df[interactions_df["user_id"] == option_user_id][
                    ["item_id", "rating"]
                ],
            )
            st.markdown("## Books the user read")

        for ind in temp_rec.index:
            col1_1, col1_2 = st.columns([0.6, 0.4])
            with col1_1:
                st.markdown(f"#### {temp_rec.loc[ind, 'Title']}")
                st.write(f"             Author: {temp_rec.loc[ind, 'Author']}")
                st.write(
                    f"             Publication Year: {temp_rec.loc[ind, 'Publication Year']}"
                )
                st.write(f"             Publisher: {temp_rec.loc[ind, 'Publisher']}")

            with col1_2:
                st.image(
                    show_book_cover(temp_rec.loc[ind, "image_url"]),
                    use_column_width=True,
                )


def new_user_recommendation_tab():
    st.header("New user recommendation tab")

    new_user_id = users_df["user_id"].unique().max() + 1
    st.write("Fill information about yourself")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        option_user_age = st.selectbox(
            "Select your Age Category",
            ["0-10", "11-19", "19-30", "31-40", "41-60", "61+", "Age_Unknown"],
            placeholder="Select your age",
        )
    with col2:
        option_user_country = st.selectbox(
            "Select your Country",
            full_interactions_df["Country"].value_counts().index.values.tolist(),
            placeholder="Select your country",
        )

    st.write()
    st.write()
    st.write("Choose number of books you have read and rated")
    number_of_books_read = st.number_input("Number of books", min_value=1, max_value=20)

    books_read_ids = []
    ratings_of_read_books = []

    with st.form("Fill your books and ratings"):
        for i in range(number_of_books_read):
            book_col, rating_col = st.columns([0.8, 0.2])
            with book_col:
                book_read = st.selectbox(
                    "Select a book",
                    items_df["Title"].sort_values().unique(),
                    placeholder="Select a book",
                    index=random.randint(0, len(items_df["Title"].unique()) - 1),
                    key=f"tab5_book_key_{i}",
                    # disabled=True
                )
                book_read_id, author_temp = items_df[items_df["Title"] == book_read][
                    ["item_id", "Author"]
                ].values[0]
                books_read_ids.append(book_read_id)
                st.write(author_temp)

            with rating_col:
                rating_of_read_books = st.selectbox(
                    "Select a rating",
                    list(range(1, 11)),
                    placeholder="Select a rating",
                    index=random.randint(0, 9),
                    key=f"tab5_rating_key_{i}",
                    # disabled=True
                )
                ratings_of_read_books.append(rating_of_read_books)

        submitted = st.form_submit_button("Submit")

        if submitted:
            user_temp_df = pd.DataFrame(
                {
                    "user_id": [new_user_id] * number_of_books_read,
                    "item_id": books_read_ids,
                    "rating": ratings_of_read_books,
                }
            )
            interactions_df_temp = pd.concat([interactions_df, user_temp_df])

            (
                dataset_temp,
                user_ids_buffered_temp,
                item_ids_buffered_temp,
                interactions_temp,
                weights_temp,
            ) = create_lightfm_dataset(interactions_df_temp)

            with st.spinner("Getting predictions..."):
                model_temp = train_lightfm_model(weights=weights_temp)

            already_rated_inds, recommended_inds = get_recommendation(
                model_temp,
                dataset_temp,
                interactions_df_temp,
                new_user_id,
                item_ids_buffered_temp,
                10,
            )
            temp_rec = items_df[items_df["item_id"].isin(recommended_inds)]
            st.markdown("## Recommendations")
            for ind in temp_rec.index:
                col1_1, col1_2 = st.columns([0.6, 0.4])
                with col1_1:
                    st.markdown(f"#### {temp_rec.loc[ind, 'Title']}")
                    st.write(f"             Author: {temp_rec.loc[ind, 'Author']}")
                    st.write(
                        f"             Publication Year: {temp_rec.loc[ind, 'Publication Year']}"
                    )
                    st.write(
                        f"             Publisher: {temp_rec.loc[ind, 'Publisher']}"
                    )

                with col1_2:
                    st.image(
                        show_book_cover(temp_rec.loc[ind, "image_url"]),
                        use_column_width=True,
                    )


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Project Description",
        "Map plot",
        "The most popular books",
        "Recommendations",
        "New user recommendations",
    ]
)

with tab1:
    project_description_tab()
with tab2:
    map_tab()
with tab3:
    popular_books_tab()
with tab4:
    recommendation_tab()
with tab5:
    new_user_recommendation_tab()
