# Recipe for Success
## Summary of Findings
### Framing the problem
We want to be able to predict the ratings of recipes on the website. In our own experience when we search up a new dish that we want to learn, we end up searching the recipe for that and we often want to compare recipes so that we get the best one, and the rating we leave on the website (if we leave any) is reflective of how the recipes rank in comparison to other recipes of the same dish from a different website. Therefore, we will choose the ratings as our target variable because from the website owner's point of view, this might be the most important metric that determines the success of food.com in comparison to its competitors.

Ratings in the dataset given to us are an ordinal categorical variable, where there are only 5 options from 1-5. Based on that we will create a new variable "high_rated" which will read 1 for if the average rating for the recipe was greater than or equal to 3 and 0 for if it was lower than 3. We will use this new variable, *"high rated"* as our **response variable** since this will help answer the question <u>"What features of a recipe predict whether or not the recipe will be high rated?"</u> which will be informative in the decision making process of the owner of the website when they decide whether or not to accept a new recipe for publishing on their site. Therefore, this makes the prediction problem one of **classification**, that is specifically, we determine based on some features whether or not a recipe will be high rated.

On defining this problem, we want to make sure we only include variables that are reflective of the data generating process. As such, we cannot use the column on reviews in predicting the rating because in the data generating process the reviews are left after the rating (if they are left at all) and so we cannot train our model or build a model based on a feature that is not available at prediction time.

Thus, the metric that we will use to evaluate our model is going to be the **F1 score** since we want to make sure we balance out both precision and recall in our model, ensuring that the website owner minimizes both false negatives (i.e. a situation where the owner rejects a good recipe that would have received a high rating) and false positives (i.e. a situation where the owner accepts a bad recipe thinking it will give a good rating). Both scenarios would be detrimental for the owner and the success of the website so *it's important we use the harmonic mean of precision and recall*, and therefore the F1 score is the best way forward for the evaluation metric. Moreover, we chose the F1 score over accuracy since *our data has a case of severe class imbalance* where there are more high rated recipes than low rated recipes as is reflected by the .valuecounts() operation on our average ratings column. That is, doing the operation (final_merged['average rating'] > 3).value_counts() gives us a series where there are 226197 "True" values (i.e. high rated recipes) and 8232 "False" values (i.e. low rated recipes).


### Baseline Model
We added the **calories**, **minutes**, and **n_steps** <u>quantitative</u> features to our prediction model. We decided to add **calories** feature so that we can include the calories as a part of our prediction model. This is because when comparing different recipes, much of the time the ingredients for the same recipe have little variation. For many, checking how *much* those little variations differ can often easily be identified by evaluating the calories. Thus, this is a crucial part to include in our prediction model, as it can help us get a better gauge of the general healthiness of a food item (or for any other reason for counting calories, no judgement!). We did this by taking the data out of the list and extracting the first element of said list, as it's sorted from most to least calories. It is then used on all values in the 'nutrition' column with FunctionTransformer(), ready for preprocessing.

**Minutes** feature was added because it's another likely variable that people will look at when evaluating which recipe to choose. Over time, people have become gravitating naturally towards more activities that require less time. The quicker one can complete the recipe the better. We utilized minutes by standardizing it with StandardScaler() in order to reduce any influence of potential outliers, as minutes is more subjective to whoever is using the recipe.

Lastly, the **N_steps** feature was added for a similar reason. The more steps a recipe has (regardless of how the steps are established), the more unlikely someone is to choose it. The converse is also true, where the shorter amount of steps can yield to someone being more likely to favor a recipe over another. We used a FunctionTransformer() on this column as well.Thus, we utilize these three quantitative features for our prediction model.

Side note: it's interesting that the *reason* why we care so much about how much/little of the activity is in order to choose it is rather interesting. It makes you wonder what this says about society (or at least the demographic of people who use recipes).

The model we chose is the Decision Tree Classifier with a max depth of 40.

Upon running our code, the performance of our model is overfitting to the training data with a 99.5% accuracy. Although this is highly accurate, our testing data received an 89% accuracy. This is still relatively high, but could be improved.

From our insides of our baseline model, it looks like we need to regularize the tree depth since that leads to overfitting to the training data. After, we'll need to incorporate more features. Such as the values in the nutrition value column in order to improve the model there.
### Final Model
For our final model, we


### Fairness Analysis
We noticed earlier in our data analysis that there was a tag in our dataset in the 'tags' column on whether the recipe was "north american". In our fairness analysis we are going to examine whether our model is equally likely to predict whether or not a recipe is 5-star rated for North American recipes as well as recipes that do not have the tag "north american". This is to ensure that our model isn't biased towards any particular region because all food is good food no matter where it comes from!! Therefore, our null and alternate hypotheses are as follows:

**Null Hypothesis**: Our model is fair. Its recall for recipes tagged "north american" and and those without the tag "north american" are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: Our model is unfair. Its recall is higher than it’s recall for recipes without the tag “North American”

To do this, we created a bar plot to verify what our false negative rates are:



What this graph tells us is that non north American recipes have a higher false negative rate. However, is this difference statistically significant? We performed a permutation test to verify with difference of means being our test statistic and a significance level of 0.05.

### Permutation Test Conclusion

After performing our permutation test, we concluded with a p-value of 0.07, which was greater than our significance level of 0.05. Thus, we fail to reject our null hypothesis. This means that we cannot say our model is not fair to recipes that don't have the tags North American on them. We also cannot claim that the recall is not roughly the same for recipes with and without the tag "north american", and differences could be due to random chance.
