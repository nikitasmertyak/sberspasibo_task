# sberspasibo_task

## The Goal:
The online store has accumulated data on the interaction of buyers with goods for several months. The goal is to recommend a product that will cause the buyer to interact with it.

## Introduction:
1. The goal is to recommend a list of 10 potentially relevant products to each customer.
2. To assess the quality of recommendations, the MAP@10 metric is used
3. You can use any recommender system algorithm (collaborative filtering, content-based, hybrid, etc.). A plus would be a demonstration of the use of several algorithms at once.
4. To split the dataset into train and test, use a random split 80/20
5. The final result is a notebook in google colab with conclusions on choosing a model.

## Data Description:
All data is stored in CSV files separated by ";".

**interactions.csv** - the file stores data on the interaction of goods and buyers. Among the data there are "cold" goods and buyers. The row column stores the customer identifiers. In the col column are product identifiers. In the data column - the value of the interaction.

Product data

**item_asset.csv** - the file stores the qualitative characteristics of the product. row - item identifier, data - characteristic value. col - serial number of the feature when uploading data (does not make sense, you can get rid of this column)

**item_price.csv** - the file stores the price of the item (already normalized). row - product identifier, data - normalized price value. col - serial number of the feature when uploading data (does not make sense, you can get rid of this column)

**item_subclass.csv** - the file stores the values of the categories to which the product belongs. row - product identifier, col - category number, data - attribute of relation to the category

User data

**user_age.csv** - the file stores data on the age of users. row - user identifier, data - age value (already normalized), col - feature serial number when uploading data (does not make sense, you can get rid of this column)

**user_region.csv** - file stores one-hot encoded user region values. row - user ID, col - number of one-hot feature of the region, data - feature of the region.

## Solution
Five methods were implemented as algorithms for solving the problem:

## 1.Popularity model
A common (and usually hard-to-beat) baseline approach is the Popularity model. This model is not actually personalized - it simply recommends to a user the most popular items that the user has not previously consumed. As the popularity accounts for the "wisdom of the crowds", it usually provides good recommendations, generally interesting for most people.

## 2. SVDModel
Singular Value Decomposition is a mathematical technique widely used in various Science and Engineering applications. It aims at decomposing a rectangular matrix $A$ (of size $m$ by $n$) into a product of three matrices: $A = UDV^T$ where $V^T$ is the transpose of $V$. Here, the matrix $U$ is of size $m$ by $m$, matrix $D$ is of size $m$ by $n$, and matrix $V$ is of size $n$ by $n$. Highly efficient algorithm and implementation exists to achieve it.

## 3. Collaborative Filtering Model
Memory-based: This approach uses the memory of previous users interactions to compute users similarities based on items they've interacted (user-based approach) or compute items similarities based on the users that have interacted with them (item-based approach).
A typical example of this approach is User Neighbourhood-based CF, in which the top-N similar users (usually computed using Pearson correlation) for a user are selected and used to recommend items those similar users liked, but the current user have not interacted yet. This approach is very simple to implement, but usually do not scale well for many users

## 4.Content Based Recommender
Each object will characterize a user-item pair and contain attributes that
describe both the user and the product. In addition, signs
can describe the whole couple itself.We will train the classifier for
interaction, and it needs negative examples. Let's add random missing
interactions as negative ones.Note that the model evaluates each pair
of potential interactions, which means that it is necessary to prepare
a sample from all possible pairs of users and articles.

## 5. Factorization Machine model
The generalization of matrix decompositions — factorization machines
that can work with content information.

## Comparison
![image](https://github.com/nikitasmertyak/sberspasibo_task/blob/main/comparison_models_plot.png)
