"""
Preparing data for recsys
"""

import pandas as pd


class RecSysData:
    """Data loader for user-item interactions.
    Args:
        dir_path : str
        Path to ".csv" file with user-item interaction data.
    """

    def __init__(self, dir_path: str):
        """
        Args:
            dir_path: path to directory with data
        """
        
        self.dir_path = dir_path

    def load_interactions(self) -> pd.DataFrame:
        """Load user-item interaction data as DataFrame.
        Returns:
            interactions_df: pd.DataFrame
            Two-dimensional data structure with labeled axes.
        """

        interactions_df = pd.read_csv(self.dir_path)
        assert 'row' in interactions_df.columns
        assert 'col' in interactions_df.columns
        assert 'data' in interactions_df.columns
        interactions_df.rename(columns={'row': 'personId',
                                        'col': 'contentId',
                                        'data': 'result'}, inplace=True)

        return interactions_df

    @staticmethod
    def _merge_items_data(item_asset: pd.DataFrame, item_price: pd.DataFrame,
                          item_subclass: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            item_asset: pd.DataFrame with item asset info
            item_price: pd.DataFrame with item price info
            item_subclass: pd.DataFrame with item subclass info
        Returns:
            item_info_df: pd.DataFrame with item features
        """
        item_asset = item_asset[['row', 'data']].rename(columns={'row': 'contentId', 'data': 'characteristic value'})
        item_price = item_price[['row', 'data']].rename(columns={'row': 'contentId', 'data': 'price'})
        item_subclass = item_subclass.rename(columns={'row': 'contentId', 'col': 'category number',
                                                      'data': 'sign of relation to the category'})
        item_asset_and_prices = item_asset.merge(item_price, on='contentId', how='inner')
        item_info_df = item_asset_and_prices.merge(item_subclass, on='contentId', how='inner')

        return item_info_df

    def _get_items_data(self, item_filenames: list):
        """
        Args:
            item_filenames: list of str (file names with item features)
        Returns:
            item_info_df: pd.DataFrame with item features
        """
        item_asset = pd.read_csv(item_filenames[0])
        item_price = pd.read_csv(item_filenames[1])
        item_subclass = pd.read_csv(item_filenames[2])

        item_info_df = self._merge_items_data(item_asset, item_price, item_subclass)
        item_info_df.contentId = items.contentId.astype(str)

        return item_info_df

    @staticmethod
    def _merge_users_data(user_age: pd.DataFrame, user_region: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            user_age: pd.DataFrame with user age info
            user_region: pd.DataFrame with user region info
        Returns:
            user_info_df: pd.DataFrame with user features
        """
        user_age = user_age[['row', 'data']].rename(columns={'row': 'personId', 'data': 'age'})
        user_region = user_region.rename(columns={'row': 'personId', 'col': 'one-hot feature number of the region',
                                                  'data': 'feature_of_the_region'})
        user_info_df = user_region.merge(user_age, on='personId', how='inner')

        return user_info_df

    def _get_users_data(self, user_filenames: list) -> pd.DataFrame:
        """
        Args:
            user_filenames: list of str (file names with user features)
        Returns:
            user_info_df: pd.DataFrame with user features
        """
        user_age = pd.read_csv(user_filenames[0])
        user_region = pd.read_csv(user_filenames[1])

        user_info_df = self._merge_users_data(user_age, user_region)

        return user_info_df

    def get_interactions_and_users_data(self, user_filenames: list) -> pd.DataFrame:
        """"
        Args:
            user_filenames: list of str (file names with user features)
        Returns:
            interactions_and_users_df
        """
        interactions_df = self.load_interactions()
        user_info_df = self._get_users_data(user_filenames)
        interactions_and_users_df = pd.merge(interactions_df, user_info_df, on='personId', how='inner')
        interactions_and_users_df.personId = interactions_df.personId.astype(str)
        interactions_and_users_df.contentId = interactions_df.contentId.astype(str)

        return interactions_and_users_df
