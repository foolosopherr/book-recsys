import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from preprocessing import *
import pickle
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k
import itertools


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


def train_test_split_data(weights, test_size):
    train_weights, test_weights = random_train_test_split(
        weights, test_percentage=test_size, random_state=42
    )

    return train_weights, test_weights


def sample_hyperparameters():
    """
    Yield possible hyperparameter choices
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 128),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp"]),
            "learning_rate": np.random.choice([0.01 * i for i in range(1, 21)]),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 20),
            "random_state": [42],
        }


def hyperparameters_tuning(train, test, num_samples=50, num_threads=8, k=10):
    for i, hyperparams in enumerate(
        itertools.islice(sample_hyperparameters(), num_samples)
    ):
        num_epochs = hyperparams.pop("num_epochs")

        print(f"Model #{i+1}:")
        print()
        print("Hyperparameters:", hyperparams)
        print()

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)

        auc_score_train = auc_score(
            model,
            train,
            num_threads=num_threads,
            check_intersections=False,
        ).mean()

        auc_score_test = auc_score(
            model,
            test,
            train_interactions=train,
            num_threads=num_threads,
            check_intersections=False,
        ).mean()

        train_precision_at_10 = precision_at_k(
            model,
            train,
            k=k,
            num_threads=num_threads,
            check_intersections=False,
        ).mean()
        test_precision_at_10 = precision_at_k(
            model,
            test,
            train_interactions=train,
            k=k,
            num_threads=num_threads,
            check_intersections=False,
        ).mean()

        print(f"Train AUC {auc_score_train:.5f}, Test AUC {auc_score_test:.5f}")
        print(
            f"Train Precision@10 {train_precision_at_10:.5f}, Test Precision@10 {test_precision_at_10:.5f}"
        )
        print()
        print("=" * 40)
        print()
