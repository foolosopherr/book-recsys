import pycountry
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import requests



def create_full_interactions_df(interactions_df, users_df, items_df):
    full_interactions_df = pd.merge(interactions_df, users_df, on="user_id", how="left")
    full_interactions_df = pd.merge(
        full_interactions_df, items_df, on="item_id", how="left"
    )

    input_countries = full_interactions_df["Country"].values

    countries = {}
    for country in pycountry.countries:
        countries[country.name] = country.alpha_3

    full_interactions_df["Country Code"] = full_interactions_df["Country"].apply(
        lambda x: countries.get(x, "Unknown code")
    )

    return full_interactions_df


def plot_map(full_interactions_df, agg_function):
    kek = (
        full_interactions_df.groupby("Country Code")["rating"]
        .agg(agg_function)
        .reset_index()
    )
    if agg_function == "count":
        fig = px.choropleth(
            kek,
            locations="Country Code",
            color=np.log(kek["rating"]),
            hover_name="Country Code",
            projection="natural earth",
            locationmode="ISO-3",
            color_continuous_scale='PuBu'
        )

    else:
        fig = px.choropleth(
            kek,
            locations="Country Code",
            color=kek["rating"],
            hover_name="Country Code",
            projection="natural earth",
            locationmode="ISO-3",
            color_continuous_scale='PuBu'
        )
    return fig


def popular_books(df, n=10, age_filter=None, country_filter=None):
    if age_filter:
        df = df[df["Age Category"] == age_filter]
    if country_filter:
        df = df[df["Country"] == country_filter]

    rating_count = df.groupby("Title").count()["rating"].reset_index()
    rating_count.rename(columns={"rating": "NumberOfVotes"}, inplace=True)

    rating_average = df.groupby("Title")["rating"].mean().reset_index()
    rating_average.rename(columns={"rating": "AverageRatings"}, inplace=True)

    popularBooks = rating_count.merge(rating_average, on="Title")

    if popularBooks.shape[0] == 0:
        return 'error'

    def weighted_rate(x):
        v = x["NumberOfVotes"]
        R = x["AverageRatings"]

        return ((v * R) + (m * C)) / (v + m)

    C = popularBooks["AverageRatings"].mean()
    m = popularBooks["NumberOfVotes"].quantile(0.90)

    # popularBooks = popularBooks[popularBooks["NumberOfVotes"] >= 50]
    popularBooks["Popularity"] = popularBooks.apply(weighted_rate, axis=1)
    popularBooks = popularBooks.sort_values(by="Popularity", ascending=False)
    popularBooks = popularBooks.drop_duplicates(subset=['Title'], keep='first')
    popularBooks = pd.merge(popularBooks, df.drop_duplicates(subset=['Title', 'Author'], keep='first')[['Title', 'Author', 'Publication Year', 'Publisher', 'image_url']], on='Title', how='left')

    return (
        popularBooks[["Title", "NumberOfVotes", "AverageRatings", "Popularity", 'Title', 'Author', 'Publication Year', 'Publisher', 'image_url']]
        .reset_index(drop=True)
        .head(n)
    )


def show_book_cover(url):
    headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    img = Image.open(requests.get(url,stream=True, headers=headers).raw)
    return img
