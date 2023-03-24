# Recipe for Success
## Summary of Findings
### Problem Identification
We want to be able to predict the ratings of recipes on the website. In our own experience when we search up a new dish that we want to learn, we end up searching the recipe for that and we often want to compare recipes so that we get the best one, and the rating we leave on the website (if we leave any) is reflective of how the recipes rank in comparison to other recipes of the same dish from a different website. Therefore, we will choose the ratings as our target variable because from the website owner's point of view, this might be the most important metric that determines the success of food.com in comparison to its competitors.

Ratings in the dataset given to us are an ordinal categorical variable, where there are only 5 options from 1-5. Based on that we will create a new variable "high_rated" which will read 1 for if the average rating for the recipe was greater than or equal to 3 and 0 for if it was lower than 3. We will use this new variable, "high rated" as our response variable since this will help answer the question "What features of a recipe predict whether or not the recipe will be high rated?" which will be informative in the decision making process of the owner of the website when they decide whether or not to accept a new recipe for publishing on their site. Therefore, this makes the prediction problem one of classification, that is specifically, we determine based on some features whether or not a recipe will be high rated.

On defining this problem, we want to make sure we only include variables that are reflective of the data generating process. As such, we cannot use the column on reviews in predicting the rating because in the data generating process the reviews are left after the rating (if they are left at all) and so we cannot train our model or build a model based on a feature that is not available at prediction time.

Thus, the metric that we will use to evaluate my model is going to be the F1 score since we want to make sure we balance out both precision and recall in our model, ensuring that the website owner minimizes both false negatives (i.e. a situation where the owner rejects a good recipe that would have received a high rating) and false positives (i.e. a situation where the owner accepts a bad recipe thinking it will give a good rating). Both scenarios would be detrimental for the owner and the success of the website so it's important we use the harmonic mean of precision and recall, and therefore the F1 score is the best way forward for the evaluation metric. Moreover, we chose the F1 score over accuracy since our data has a case of severe class imbalance where there are more high rated recipes than low rated recipes as is reflected by the .valuecounts() operation on our average ratings column. That is, doing the operation (final_merged['average rating'] > 3).value_counts() gives us a series where there are 226197 "True" values (i.e. high rated recipes) and 8232 "False" values (i.e. low rated recipes).

### Baseline Model
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
