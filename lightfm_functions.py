import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from preprocessing import *
import pickle


def get_clean_dataframes(path="data/"):
    interactions_df = pd.read_csv(
        path + "interactions_clean.csv", on_bad_lines="skip", encoding="latin-1"
    )

    users_df = pd.read_csv(
        path + "users_clean.csv", encoding="latin-1", on_bad_lines="skip"
    )
    items_df = pd.read_csv(
        path + "items_clean.csv", encoding="latin-1", on_bad_lines="skip"
    )

    return interactions_df, items_df, users_df


def create_lightfm_dataset(interactions_df):
    dataset = Dataset()

    user_ids_buffered = interactions_df["user_id"].unique()
    item_ids_buffered = interactions_df["item_id"].unique()

    dataset.fit(users=user_ids_buffered, items=item_ids_buffered)

    (interactions, weights) = dataset.build_interactions(
        (
            (k, v, w)
            for k, v, w in zip(
                list(interactions_df["user_id"].values),
                list(interactions_df["item_id"].values),
                list(interactions_df["rating"].values),
            )
        )
    )

    return dataset, user_ids_buffered, item_ids_buffered, interactions, weights


def train_lightfm_model(weights, to_save=False, path="data/"):
    model = LightFM(no_components=110, learning_rate=0.025, loss="warp")

    model.fit(
        interactions=weights,
        verbose=True,
        epochs=15,
        num_threads=20,
    )

    if to_save:
        with open(path + "lightfm_model.pickle", "wb") as handle:
            pickle.dump(model, handle)

    return model


def get_recommendation(
    model, dataset, interactions_df, user_id, item_ids_buffered, top_n
):
    scores = model.predict(
        dataset.mapping()[0][user_id], np.arange(dataset.interactions_shape()[1])
    )

    recommended_inds = item_ids_buffered[np.argsort(-scores)][:100]

    already_rated_inds = interactions_df[interactions_df["user_id"] == user_id][
        "item_id"
    ].values
    recommended_inds = [i for i in recommended_inds if i not in already_rated_inds][
        :top_n
    ]

    return already_rated_inds, recommended_inds
