# Recipe for Success
## Summary of Findings
### Problem Identification
We want to be able to predict the ratings of recipes on the website. In our own experience when we search up a new dish that we want to learn, we end up searching the recipe for that and we often want to compare recipes so that we get the best one, and the rating we leave on the website (if we leave any) is reflective of how the recipes rank in comparison to other recipes of the same dish from a different website. Therefore, we will choose the ratings as our target variable because from the website owner's point of view, this might be the most important metric that determines the success of food.com in comparison to its competitors.

Ratings in the dataset given to us are an ordinal categorical variable, where there are only 5 options from 1-5. Based on that we will create a new variable "high_rated" which will read 1 for if the average rating for the recipe was greater than or equal to 3 and 0 for if it was lower than 3. We will use this new variable, *"high rated"* as our **response variable** since this will help answer the question <u>"What features of a recipe predict whether or not the recipe will be high rated?"</u> which will be informative in the decision making process of the owner of the website when they decide whether or not to accept a new recipe for publishing on their site. Therefore, this makes the prediction problem one of **classification**, that is specifically, we determine based on some features whether or not a recipe will be high rated.

On defining this problem, we want to make sure we only include variables that are reflective of the data generating process. As such, we cannot use the column on reviews in predicting the rating because in the data generating process the reviews are left after the rating (if they are left at all) and so we cannot train our model or build a model based on a feature that is not available at prediction time.

Thus, the metric that we will use to evaluate our model is going to be the **F1 score** since we want to make sure we balance out both precision and recall in our model, ensuring that the website owner minimizes both false negatives (i.e. a situation where the owner rejects a good recipe that would have received a high rating) and false positives (i.e. a situation where the owner accepts a bad recipe thinking it will give a good rating). Both scenarios would be detrimental for the owner and the success of the website so *it's important we use the harmonic mean of precision and recall*, and therefore the F1 score is the best way forward for the evaluation metric. Moreover, we chose the F1 score over accuracy since *our data has a case of severe class imbalance* where there are more high rated recipes than low rated recipes as is reflected by the .valuecounts() operation on our average ratings column. That is, doing the operation (final_merged['average rating'] > 3).value_counts() gives us a series where there are 226197 "True" values (i.e. high rated recipes) and 8232 "False" values (i.e. low rated recipes).

## Code
```python
# Imports
import pandas as pd
import numpy as np
import os

import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'
TEMPLATE = 'seaborn'

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
```
### Framing the Problem
```
recipes = pd.read_csv("RAW_recipes.csv")
interactions = pd.read_csv("RAW_interactions.csv")
merged = recipes.merge(interactions, how = 'left', left_on = 'id', right_on = 'recipe_id')
merged['rating'] = merged['rating'].replace(0, np.nan)
avg_ratings = merged.groupby('id').mean()[['rating']]
final_merged = merged.merge(avg_ratings, how = 'left', left_on = 'id', right_index = True)
final_merged = final_merged.drop(columns = ['user_id', 'contributor_id', 'submitted', 'date', 'review', 'recipe_id'])
final_merged = final_merged.rename(columns = {'rating_y' : 'average rating'})
final_merged = final_merged.dropna(subset = ['rating_x']).set_index('id')
final_merged['rated 5'] = final_merged['average rating'] == 5
```
```
X_train, X_test, y_train, y_test = train_test_split(final_merged.iloc[:,:-1], final_merged['rated 5'], test_size = 0.2 )
```
### Baseline Model
```
# Feature 1 - Nutrition Values
def extract_calories(df):
    ser = df['nutrition'].str.strip("[]").str.split(",").str[0]
    return pd.DataFrame(ser)
```
```
preproc = ColumnTransformer([
    ('std', StandardScaler(), ['minutes']),
    ('calories', FunctionTransformer(extract_calories), ['nutrition']),
    ('num of steps', FunctionTransformer(lambda x: x), ['n_steps'])
], remainder = 'drop')
pl = Pipeline([
    ('preproc', preproc),
    ('dt', DecisionTreeClassifier())
])
pl.fit(X_train, y_train)
```
```
pl.score(X_train, y_train) # High accuracy for training data
```
```
pl.score(X_test, y_test) # Also pretty high accuracy for testing data.
```
```
pl['dt'].tree_.max_depth
```
```
f1_score(y_test, pl.predict(X_test))
```
From our insides of our baseline model, it looks like we need to regularize the tree depth since that leads to overfitting to the training data. After, we'll need to incorporate more features. Such as the values in the nutrition value column in order to improve the model there.
### Final Model
