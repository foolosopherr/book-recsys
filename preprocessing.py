import pandas as pd
import numpy as np
import re


def change_country_name(country_name):
    return re.sub(r"[^a-zA-Z\s]", "", country_name.lower().capitalize())


def replace_non_letters(sentence):
    sentence = re.sub(r"[!#$%&/()*+-/:;<=>?@[\]^_`{|}~]", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def preprocess_users(users_df):
    users_df.columns = ["userID", "Location", "Age"]

    # Age
    age_bins = [0, 10, 18, 30, 40, 60, float("inf")]
    age_labels = ["0-10", "11-19", "19-30", "31-40", "41-60", "61+"]

    users_df["Age Category"] = pd.cut(users_df["Age"], bins=age_bins, labels=age_labels)

    users_df["Age Category"].cat.add_categories(["Age_Unknown"], inplace=True)
    users_df["Age Category"].fillna("Age_Unknown", inplace=True)

    # Country
    users_df["Country"] = (
        users_df["Location"].str.split(",").str[-1].str.strip().str.capitalize()
    )

    users_df["Country"].replace("Na", "", inplace=True)
    users_df["Country"].replace("None", "", inplace=True)
    users_df["Country"].replace("United states", "Usa", inplace=True)
    users_df["Country"].replace("United arab emirates", "Uae", inplace=True)
    users_df["Country"].replace("La france", "France", inplace=True)
    users_df["Country"].replace("Italia", "Italy", inplace=True)
    users_df["Country"].replace("Espaa", "Spain", inplace=True)
    users_df["Country"].replace("Uk", "United kingdom", inplace=True)
    users_df["Country"].replace("England", "United kingdom", inplace=True)
    users_df["Country"].replace("Ireland", "United kingdom", inplace=True)
    users_df["Country"].replace("Scotland", "United kingdom", inplace=True)
    users_df["Country"].replace("Wales", "United kingdom", inplace=True)

    users_df["Country"].replace("Deutschland", "Germany", inplace=True)
    users_df["Country"].replace("Urugua", "Uruguay", inplace=True)
    users_df["Country"].replace("Litalia", "Italy", inplace=True)
    users_df["Country"].replace("Jersey", "Usa", inplace=True)

    users_df["Country"] = users_df["Country"].str.title()

    users_df["Country"].replace("Uae", "UAE", inplace=True)
    users_df["Country"].replace("Usa", "USA", inplace=True)

    country_counts = users_df["Country"].value_counts()

    popular_countries = country_counts[(country_counts <= 10)].index

    users_df.loc[
        users_df["Country"].isin(popular_countries), "Country"
    ] = "Country_Unknown"

    users_df["Country"].replace("L`Italia", "Italy", inplace=True)
    users_df['Country'].replace('USA', 'United States', inplace=True)
    users_df['Country'].replace('Iran', 'Iran, Islamic Republic of', inplace=True)
    users_df['Country'].replace('Taiwan', 'Taiwan, Province of China', inplace=True)
    users_df['Country'].replace('Trinidad And Tobago', 'Trinidad and Tobago', inplace=True)
    users_df['Country'].replace('South Korea', 'Korea, Republic of', inplace=True)
    users_df['Country'].replace('Czech Republic', 'Czechia', inplace=True)
    users_df['Country'].replace('UAE', 'United Arab Emirates', inplace=True)
    users_df['Country'].replace('Euskal Herria', 'Country_Unknown', inplace=True)

    users_df['Country'].replace('Yugoslavia', 'Country_Unknown', inplace=True)
    users_df['Country'].replace('Euskal Herria', 'Country_Unknown', inplace=True)
    users_df['Country'].replace('Burma', 'Myanmar', inplace=True)

    users_df['Country'].replace('Venezuela', 'Venezuela, Bolivarian Republic of', inplace=True)
    users_df['Country'].replace('Macedonia', 'Montenegro', inplace=True)
    users_df['Country'].replace('Russia', 'Russian Federation', inplace=True)
    users_df['Country'].replace('Caribbean Sea', 'Country_Unknown', inplace=True)


    users_df['Country'] = users_df['Country'].apply(lambda x: re.sub(r'"', '', x))

    users_df["Country"].replace('N/A', "Country_Unknown", inplace=True)

    users_df["Country"].replace("", "Country_Unknown", inplace=True)
    users_df = users_df[["userID", "Age Category", "Country"]]
    users_df.columns = ["user_id", "Age Category", "Country"]

    return users_df


def preprocess_items(items_df):
    books_urls = items_df[["ISBN", "Image-URL-M"]]

    items_df = items_df.iloc[:, :-3]
    items_df.columns = ["ISBN", "Title", "Author", "Publication Year", "Publisher"]

    # Changing errors
    items_df.loc[
        209538, "Title"
    ] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
    items_df.loc[209538, "Author"] = "Michael Teitelbaum"
    items_df.loc[209538, "Publication Year"] = 2000
    items_df.loc[209538, "Publisher"] = "DK Publishing Inc"

    items_df.loc[
        221678, "Title"
    ] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
    items_df.loc[221678, "Author"] = "James Buckley"
    items_df.loc[221678, "Publication Year"] = 2000
    items_df.loc[221678, "Publisher"] = "DK Publishing Inc"

    items_df.loc[220731, "Title"] = "Peuple du ciel, suivi de 'Les Bergers"
    items_df.loc[220731, "Author"] = "Jean-Marie Gustave Le Clezio"
    items_df.loc[220731, "Publication Year"] = 2003
    items_df.loc[220731, "Publisher"] = "Gallimard"

    items_df["Publication Year"] = items_df["Publication Year"].astype(int)
    items_df.loc[items_df["Author"].isna(), "Author"] = "Author_Unknown"
    items_df.loc[items_df["Publisher"].isna(), "Publisher"] = "Publisher_Unknown"
    items_df["Author"] = items_df["Author"].apply(
        lambda x: replace_non_letters(x).title()
    )

    items_df["Title"] = items_df["Title"].apply(
        lambda x: replace_non_letters(x).title()
    )
    items_df["Publisher"] = items_df["Publisher"].apply(
        lambda x: replace_non_letters(x).title()
    )

    items_df.columns = ["item_id", "Title", "Author", "Publication Year", "Publisher"]
    books_urls.columns = ["item_id", "image_url"]

    return items_df, books_urls


def preprocess_interactions(interactions_df, items_df, users_df, books_urls, min_count_author, min_count_user_ints):
    interactions_df.columns = ['user_id', 'item_id', 'rating']

    authors_to_keep = items_df["Author"].value_counts()
    authors_to_keep = authors_to_keep[authors_to_keep >= min_count_author].index.values

    items_df["Author"] = items_df["Author"].apply(
        lambda x: x if x in authors_to_keep else "Noname"
    )
    interactions_df = interactions_df[interactions_df["rating"] != 0]

    counts = interactions_df["item_id"].value_counts()
    interactions_df = interactions_df[
        interactions_df["item_id"].isin(counts[counts >= min_count_user_ints].index)
    ]

    users_df = users_df[users_df["user_id"].isin(interactions_df["user_id"].unique())]
    items_df = items_df[items_df["item_id"].isin(interactions_df["item_id"].unique())]

    interactions_df = interactions_df[
        interactions_df["user_id"].isin(users_df["user_id"].unique())
    ]
    interactions_df = interactions_df[
        interactions_df["item_id"].isin(items_df["item_id"].unique())
    ]

    books_urls = books_urls[books_urls["item_id"].isin(items_df["item_id"].values)]

    return interactions_df, items_df, users_df, books_urls


def preprocess_all(interactions_df, items_df, users_df, min_count_author, min_count_user_ints):
    users_df = preprocess_users(users_df)
    items_df, books_urls = preprocess_items(items_df)
    interactions_df, items_df, users_df, books_urls = preprocess_interactions(
        interactions_df, items_df, users_df, books_urls, min_count_author, min_count_user_ints
    )

    items_df = pd.merge(items_df, books_urls, on='item_id', how='left')

    return interactions_df, items_df, users_df
