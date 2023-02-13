"""
An example of using models.
"""

from recsys_data_preprocessing import RecSysData
from metrics import calc_metrics
from sklearn.model_selection import train_test_split
from model import *
from prepare_data import *


def main():
    """Default model training."""

    rec = RecSysData("./data/interactions.csv")
    interactions_df = rec.get_interactions_and_users_data(['./data/user_age.csv', './data/user_region.csv'])
    item_info_df = rec.get_items_data(['./data/item_asset.csv', './data/item_price.csv', './data/item_subclass.csv'])

    # reduce the amount of data 
    interactions_full_df = collapse_all_actions_with_one_product_into_one_interaction(interactions_df, 50)
    interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.2,
                                                                   random_state=42)

    interactions = prepare_data_for_convenient_measuring_quality(interactions_train_df, interactions_test_df)

    baseline_model = Baseline(interactions)
    baseline_model.fit(interactions_train_df)
    baseline_model.predict()

    svd_model = SVDModel(interactions)
    svd_model.fit(interactions_train_df)
    svd_model.predict()

    collaborative_filtering_model = CollaborativeFilteringModel(interactions)
    collaborative_filtering_model.fit(interactions_train_df)
    collaborative_filtering_model.predict()

    X_train, y_train, test = prepare_data_for_content_based_recommender(interactions, interactions_train_df,
                                                                        item_info_df)
    content_based_recommender_model = ContentBasedRecommender(interactions)
    content_based_recommender_model.fit(X_train, y_train)
    content_based_recommender_model.predict(test)
    
    train_features, test_features, y_train = prepare_data_for_factorization_machines(interactions_train_df,
                                                                                     interactions)
    factorization_machines_model = FactorizationMachines(interactions)
    factorization_machines_model.fit(train_features, y_train, interactions_train_df)
    factorization_machines_model.predict(test_features)
    
    print('Baseline score: ', calc_metrics('prediction_popular', interactions))
    print('Collaborative Filtering score: ', calc_metrics('prediction_user_based', interactions))
    print('SVD score: ', calc_metrics('prediction_svd', interactions))
    print('Content Based Recommender score: ', calc_metrics('prediction_content', interactions))
    print('FactorizationMachines score: ', calc_metrics('fm_prediction', interactions))


if __name__ == '__main__':
    main()
