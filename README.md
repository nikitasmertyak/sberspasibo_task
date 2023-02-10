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
