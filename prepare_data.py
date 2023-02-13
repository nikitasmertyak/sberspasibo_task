"""
Functions for working with dataset for different models
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm_notebook


def users_with_enough_interactions(interactions_df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Reminder systems are prone to the cold start problem. We will work only with those users who interacted with at
    least threshold products. Leave only such users.

    Args:
        interactions_df: DataFrame with interactions
        threshold: threshold for numbers of interactions for each user

    Returns:
        users_with_enough_interactions_df: DataFrame with interactions where users with
            enough interactions 
    """
    users_interactions_count_df = (
        interactions_df
        .groupby(['personId', 'contentId'])
        .first()
        .reset_index()
        .groupby('personId').size())

    users_with_enough_interactions_df = \
        users_interactions_count_df[users_interactions_count_df >= threshold].reset_index()[['personId']]

    return users_with_enough_interactions_df


def leave_interactions_that_concern_only_filtered_users(interactions_df: pd.DataFrame,
                                                        threshold: int = 10) -> pd.DataFrame:
    """
    Leave only those interactions that concern only filtered users.

    Args:
        interactions_df: DataFrame with interactions
        threshold: threshold for numbers of interactions for each user

    Returns:
        interactions_from_selected_users_df: DataFrame with interactions where users with
            enough interactions in original dataset
    """
    interactions_from_selected_users_df = interactions_df.loc[np.in1d(interactions_df.personId,
                                                                      users_with_enough_interactions(interactions_df,
                                                                                                     threshold))]

    return interactions_from_selected_users_df


def collapse_all_actions_with_one_product_into_one_interaction(interactions_df: pd.DataFrame,
                                                               threshold: int = 10) -> pd.DataFrame:
    """
    In this case, each user could interact with each product more than once (at least buying the same product).
    It is proposed to "collapse" all actions into one interaction with a weight equal to the sum of interactions.
    
    Args:
        interactions_df: DataFrame with interactions
        threshold: threshold for numbers of interactions for each user

    Returns:
        interactions_full_df: DataFrame with interactions convenient for using in model
    """
    interactions_full_df = (
        leave_interactions_that_concern_only_filtered_users(interactions_df, threshold)
        .groupby(['personId', 'contentId']).result.sum()
        .reset_index().set_index(['personId', 'contentId'])
    )
    interactions_full_df = interactions_full_df.reset_index()

    return interactions_full_df


def prepare_data_for_convenient_measuring_quality(interactions_train_df: pd.DataFrame,
                                                  interactions_test_df: pd.DataFrame) -> pd.DataFrame:
    """
    For the convenience of calculating the quality, we will write the data in a format where the row corresponds to
    the user, and the columns will be true labels and predictions in the form of lists.

    Args:
        interactions_train_df: DataFrame with interactions which are in the train part
        interactions_test_df: DataFrame with interactions which are in the train part

    Returns:
        interactions: DataFrame with columns ["personId", "true_train", "true_test"]
    """
    interactions = (
        interactions_train_df
        .groupby('personId')['contentId'].agg(lambda x: list(x))
        .reset_index()
        .rename(columns={'contentId': 'true_train'})
        .set_index('personId')
    )

    interactions['true_test'] = (
        interactions_test_df
        .groupby('personId')['contentId'].agg(lambda x: list(x))
    )

    # fill gaps with empty lists
    interactions.loc[pd.isnull(interactions.true_test), 'true_test'] = [
        [''] for x in range(len(interactions.loc[pd.isnull(interactions.true_test), 'true_test']))]

    return interactions


def create_matrix_ratings(train_interactions_df: pd.DataFrame):
    """
    Matrix of interactions
    Args:
        train_interactions_df: DataFrame with interactions
    Returns:
        ratings: DataFrame with interactions between users and products
    """
    ratings = pd.pivot_table(
        train_interactions_df,
        values='result',
        index='personId',
        columns='contentId').fillna(0)

    return ratings


def prepare_data_for_content_based_recommender(interactions: pd.DataFrame,
                                               train_interactions_df: pd.DataFrame,
                                               item_info_df: pd.DataFrame):
    """
    Each object will characterize a user-item pair and contain attributes
    that describe both the user and the product. In addition, signs can
    describe the whole couple itself.
    Args:
        interactions: DataFrame with interactions
        train_interactions_df: DataFrame with interactions which are in the train part
        item_info_df: DataFrame with item information about items
    Returns:
        X_train: DataFrame for training the gradient boosting
        y_train: array, target for training part
        test: DataFrame for testing the gradient boosting
    """
    ratings = create_matrix_ratings(train_interactions_df)
    test_personId = np.repeat(interactions.index, len(ratings.columns))
    test_contentId = list(ratings.columns) * len(interactions)
    test = pd.DataFrame(
        np.array([test_personId, test_contentId]).T,
        columns=['personId', 'contentId'])

    train_interactions_df = pd.concat((
        train_interactions_df,
        test.loc[
            np.random.permutation(test.index)[
                :4 * len(train_interactions_df)]]), ignore_index=True)
    train_interactions_df.result.fillna(0, inplace=True)

    train_interactions_df = train_interactions_df.merge(item_info_df, how='left', on='contentId')

    X_train = train_interactions_df.drop(
        columns=["personId", "contentId", "result", "sign of relation to the category"])
    y_train = train_interactions_df["result"]

    test_personId = np.repeat(interactions.index, len(item_info_df))
    test_contentId = list(item_info_df.contentId) * len(interactions)
    test = pd.DataFrame(
        np.array([test_personId, test_contentId]).T,
        columns=['personId', 'contentId'])
    test = test.merge(item_info_df, how='left', on='contentId')

    return X_train, y_train, test


def prepare_data_for_factorization_machines(interactions_train_df: pd.DataFrame, interactions: pd.DataFrame):
    """
    Generate a table with features in this form, where there will be user id, item.
    """
    ratings = create_matrix_ratings(interactions_train_df)
    train_data = []
    test_data = []

    for i in tqdm_notebook(range(len(interactions_train_df))):
        features = {'personId': str(interactions_train_df.iloc[i].personId),
                    'contentId': str(interactions_train_df.iloc[i].contentId)}
        train_data.append(features)

    for i in tqdm_notebook(range(len(interactions))):
        features = {'personId': str(interactions.index[i])}
        for j in range(len(ratings.columns)):
            features['contentId'] = str(ratings.columns[j])
            test_data.append(deepcopy(features))

    dv = DictVectorizer()

    train_features = dv.fit_transform(
        train_data + list(np.random.permutation(test_data)[:100000]))
    test_features = dv.transform(test_data)
    y_train = list(interactions_train_df.result.values) + list(np.zeros(100000))

    return train_features, test_features, y_train
