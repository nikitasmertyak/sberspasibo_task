"""
Recsys models.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import numpy as np
from scipy.linalg import svd
from tqdm import tqdm_notebook
from prepare_data import create_matrix_ratings
import catboost
from pyfm import pylibfm


class RecommendationModel(ABC):
    """An abstract base class for recommendation model."""

    @abstractmethod
    def __init__(self):
        self._train_interactions: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, train_interactions):
        """Implementation of training the model should be written in this method."""

        raise NotImplementedError()

    def predict(self, k_items: int):
        """Implementation of training the model should be written in this method."""

        if k_items <= 0:
            raise ValueError(f'Expected k_items > 0.')

        raise NotImplementedError()


class Baseline(RecommendationModel):
    """The simplest recommendation model (yet quite strong!) is the one that
       recommends the most popular items.

       Args:
           interactions: Two-dimensional data structure with labeled axes.
    """

    def __init__(self, interactions) -> None:
        super().__init__()
        self.interactions = interactions
        self.popular_content = None

    def fit(self, train_interactions: pd.DataFrame):
        """Train model.

        Args:
        
            train_interactions : DataFrame, where to find user interactions with products

        Returns:

            self : returns an instance of self.
        """

        self._train_interactions = train_interactions.copy()
        self.popular_content = (
            self._train_interactions
            .groupby('contentId')
            .result.sum().reset_index()
            .sort_values('result', ascending=False)
            .contentId.values
        )

    def predict(self, k_items: int = 10):
        """Predict k items for each user.

        Args:
        
            k_items : int
                      The number of predicted items per user.
        """
        self.interactions['prediction_popular'] = (
            self.interactions.true_train
            .apply(
                lambda x:
                self.popular_content[~np.in1d(self.popular_content, x)][:k_items]
            )
        )


class SVDModel(RecommendationModel):
    """SVD recommendation model.

    Args:
    
        k : int, default=100
            The number of singular values and vectors to compute.
        interactions: Two-dimensional data structure with labeled axes.
    """

    def __init__(self, interactions, k: int = 100) -> None:

        if k <= 0:
            raise ValueError(f'Expected k > 0.')

        super().__init__()
        self.k = k
        self.U: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.ratings: Optional[np.ndarray] = None
        self.interactions = interactions

    def fit(self, train_interactions: pd.DataFrame):
        """Train model.

        Args:
        
            train_interactions : DataFrame, contains user interactions with products

        Returns:

            self : returns an instance of self.
        """

        self._train_interactions = train_interactions.copy()
        self.ratings = pd.pivot_table(
            self._train_interactions,
            values='result',
            index='personId',
            columns='contentId').fillna(0)
        self.U, self.sigma, self.V = svd(self.ratings)

        return self

    def predict(self, k_items: int = 10):
        """Predict k items for each user.

        Args:
        
            k_items : int
                      The number of predicted items per user.
        """
        self.sigma[self.k:] = 0
        Sigma = np.zeros((self.U.shape[0], self.V.shape[0]))
        Sigma[:self.U.shape[0], :self.U.shape[0]] = np.diag(self.sigma)
        new_ratings = self.U.dot(Sigma).dot(self.V)
        new_ratings = pd.DataFrame(new_ratings, index=self.ratings.index, columns=self.ratings.columns)
        predictions = []
        for personId in tqdm_notebook(self.interactions.index):
            prediction = (
                new_ratings
                .loc[personId]
                .sort_values(ascending=False)
                .index.values
            )

            predictions.append(
                list(prediction[~np.in1d(
                    prediction,
                    self.interactions.loc[personId, 'true_train'])])[:k_items])

        self.interactions['prediction_svd'] = predictions


class CollaborativeFilteringModel(RecommendationModel):
    """CollaborativeFiltering recommendation model.
       Model to recommend the items to user, using info about the part
       of user items and other users interactions.

       Args:
         alpha: measure of user similarity
         interactions: Two-dimensional data structure with labeled axes.
    """

    def __init__(self, interactions: pd.DataFrame, alpha: int = 0) -> None:

        super().__init__()
        self.interactions = interactions
        self.similarity_users = None
        self.ratings_m = None
        self.ratings: Optional[np.ndarray] = None
        self.alpha = alpha

    def fit(self, train_interactions: pd.DataFrame):
        """Train model.

        Args:
        
            train_interactions : DataFrame
                                User interactions with products for training

        Returns:

            self : returns an instance of self.
        """

        self._train_interactions = train_interactions.copy()
        self.ratings = create_matrix_ratings(self._train_interactions)
        self.ratings_m = self.ratings.values
        self.similarity_users = np.zeros((len(self.ratings_m), len(self.ratings_m)))

        for i in tqdm_notebook(range(len(self.ratings_m) - 1)):
            for j in range(i + 1, len(self.ratings_m)):

                # nonzero elements of two users
                mask_uv = (self.ratings_m[i] != 0) & (self.ratings_m[j] != 0)

                # continue if no intersection
                if np.sum(mask_uv) == 0:
                    continue

                # get nonzero elements
                ratings_v = self.ratings_m[i, mask_uv]
                ratings_u = self.ratings_m[j, mask_uv]

                # for nonzero std
                if len(np.unique(ratings_v)) < 2 or len(np.unique(ratings_u)) < 2:
                    continue

                self.similarity_users[i, j] = np.corrcoef(ratings_v, ratings_u)[0, 1]
                self.similarity_users[j, i] = self.similarity_users[i, j]

        return self

    def predict(self, k_items: int = 10):
        """Predict k items for each user.

        Args:
        
            k_items : int
                      The number of predicted items per user.
        """
        prediction_user_based = []
        for i in tqdm_notebook(range(len(self.similarity_users))):
            users_sim = self.similarity_users[i] > self.alpha
            if len(users_sim) == 0:
                prediction_user_based.append([])
            else:
                tmp_recommend = np.argsort(self.ratings_m[users_sim].sum(axis=0))[::-1]
                tmp_recommend = self.ratings.columns[tmp_recommend]
                recommend = np.array(tmp_recommend)[~np.in1d(tmp_recommend, self.interactions.iloc[i])][:k_items]
                prediction_user_based.append(list(recommend))

        self.interactions['prediction_user_based'] = prediction_user_based


class ContentBasedRecommender(RecommendationModel):
    """Implement an alternative approach to recommender systems - content models.
       Each object will characterize a user-item pair and contain attributes that
       describe both the user and the product. In addition, signs
       can describe the whole couple itself.We will train the classifier for
       interaction, and it needs negative examples. Let's add random missing
       interactions as negative ones.Note that the model evaluates each pair
       of potential interactions, which means that it is necessary to prepare
       a sample from all possible pairs of users and articles.

    Args:
           interactions: Two-dimensional data structure with labeled axes.
    """

    def __init__(self, interactions: pd.DataFrame) -> None:
        super().__init__()
        self.interactions = interactions
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.array):
        """Train model.

        Args:
        
            X_train : DataFrame
                      Contains user item features
            y_train : array
                      Contains target value

        Returns:

            self : returns an instance of self.
        """
        self.model = catboost.CatBoostClassifier()
        self.model.fit(X_train, y_train)

        return self

    def predict(self, test: pd.DataFrame, k_items: int = 10):
        """Predict k items for each user.

        Args:

            test: DataFrame
                  Using for test the model
            k_items : int
                      The number of predicted items per user.
        """
        predictions = self.model.predict_proba(test)[:, 1]
        test['predictions'] = predictions

        test = test.sort_values('predictions', ascending=False)
        predictions = test.groupby('personId')['contentId'].aggregate(list)
        tmp_predictions = []

        for personId in tqdm_notebook(self.interactions.index):
            prediction = np.array(predictions.loc[personId])

            tmp_predictions.append(
                list(prediction[~np.in1d(
                    prediction,
                    self.interactions.loc[personId, 'true_train'])])[:k_items])

        self.interactions['prediction_content'] = tmp_predictions


class FactorizationMachines(RecommendationModel):
    """ The generalization of matrix decompositions — factorization machines
        that can work with content information.

    Args:
           interactions: pd.DataFrame with true_train and true_test information
           hidden_view_size: int Hidden view size
           num_iter: int count of iterations
    """

    def __init__(self, interactions: pd.DataFrame,
                 hidden_view_size: int = 10, num_iter: int = 30) -> None:
        super().__init__()
        self.interactions = interactions
        self.hidden_view_size = hidden_view_size
        self.num_iter = num_iter
        self.fm = None
        self.ratings = None

    def fit(self, train_features: pd.DataFrame, y_train: np.array, train_interactions):
        """Train model.

        Args:
        
            train_features : DataFrame
                      Contains user - item features
            y_train: array
                    target for training
            train_interactions:

        Returns:

            self : returns an instance of self.
        """
        self.ratings = create_matrix_ratings(train_interactions)
        self.fm = pylibfm.FM(num_factors=self.hidden_view_size, num_iter=self.num_iter, task='regression')
        self.fm.fit(train_features, y_train)

        return self

    def predict(self, test_features: pd.DataFrame, k_items: int = 10):
        """Predict k items for each user.

        Args:

            test_features: DataFrame
                  Using for test the model
            k_items : int
                      The number of predicted items per user.
        """
        y_predict = self.fm.predict(test_features)
        predictions = []
        new_ratings = y_predict.reshape(self.ratings.shape)

        for i, person in enumerate(self.interactions.index):
            user_prediction = self.ratings.columns[np.argsort(new_ratings[i])[::-1]]
            predictions.append(
                user_prediction[~np.in1d(user_prediction,
                                         self.interactions.loc[person, 'true_train'])][:k_items])

        self.interactions['fm_prediction'] = predictions
