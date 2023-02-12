import pandas as pd
import numpy as np


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
