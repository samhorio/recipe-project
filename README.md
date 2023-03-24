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


### Final Model
For the final model we chose to add features from the tags column of the data. Since the tags on a website are often ways a user is directed to that particular recipe or product, we can hypothesize about the data generating process that the tags for each recipe garner a lot of user interaction and might therefore have high predictive power for classifying whether a recipe is rated 5-stars or not.

We first started by unpacking the “tags” column. So we first found a series of all the tags in the dataset containing the recipes from food.com (where each recipe has one row to itself).  We then used the value_counts() function to find the top 25 most common tags out of the 553 tags that are in the “tags” column stored in embedded list form.
Out of the all the tags in the recipes dataset, since there are 81,173 unique recipes on the website in total, we observe by looking at the top 3 tags that almost all recipes have the tags "preparation", "time-to-make" and "course". Thus we completely eliminate the possibility of using these tags as features. Moreover, we see that popular tags include "60 minutes or less", "4 hours or less", "30 minutes or less", and other time based tokens. Similarly, we also see "3-steps or less" as a tag. Since we already plan to incorporate minutes and number of steps into our model, we choose to eliminate those tags as additional features since our model is already using that information. The tags we choose to incorporate are:

1. Main ingredient,
2. dietary
3. easy,
4. occasion,
5. cuisine
6. main-dish

The final choice we made was definitely a more intuitive exercise in trying to find features that were different from each other (to prevent multicollinearity) and to capture as much information from the data as possible.

Moreover, we notice that "low in something", "low carb", "low sodium", "low-cholestrol" are popular tags in the top 25 tags too so in our model we feature engineered the nutrition values in order to incorporate these insights in our model better. While sodium and carbohydtrates are given nutritional values, we can capture "cholestrol" using saturated fats and calories.

We believe these features improved our model from the perspective of the data generating process because (1) website tags are highly interactive objects on a website and receive much user interaction. Especially in the case of recipes people usually interact with these tags to find similar recipes to the one they’re viewing at the point in time. Thus, including tags as features helps us incorporate a crucial part of the data generating process in our model. Moreover, including (2) more nutritional value data gives a more holistic understanding of the recipe itself and what the user might be judging it and rating it by, especially since the tags revealed “healthy”, “low sugar” etc. as being a classification that users make for recipes, indicating that that is something that users prioritize and would keep in mind when rating a recipe.
We first began by adding the features into the model but since Decision Tree classifiers tend to overfit and are not very good at handling with complex features, our model didn’t give us the improvement we were hoping for. Therefore we ran a grid search over three hyperparameters – max_depth, min_samples_split and criterion, most of which were aimed at trying to increase regularization so that the model generalizes better to test data. However, our grid search based classifier with even the best parameters did not perform better than our baseline model. We realized that because of the number of features that we were using, we should use a model that is more complex in its workings. We decided to use a Random Forest Classifier since instead of searching for the best feature when splitting at a node, the algorithm searches for the best feature among a random subset of features. Therefore, the model would result in greater tree diversity overall. Finally, RandomForestClassifier is by nature an ensemble method that incorporates bootstrapping so with the increased number of features, that additional hyperparameter would help in training the model well.

While on the surface our final model (accuracy score = 90.4%, f1 = 88.9%) did only a little better than our baseline model (accuracy score = 89.05%, f1 = 86.9%) we believe it captures information about the data generating process more holistically and as showed by the improved f1 score, will make lesser misclassification errors when introduced to new data.



### Fairness Analysis
We noticed earlier in our data analysis that there was a tag in our dataset in the 'tags' column on whether the recipe was "north american". In our fairness analysis we are going to examine whether our model is equally likely to predict whether or not a recipe is 5-star rated for North American recipes as well as recipes that do not have the tag "north american". This is to ensure that our model isn't biased towards any particular region because all food is good food no matter where it comes from!! Therefore, our null and alternate hypotheses are as follows:

**Null Hypothesis**: Our model is fair. Its recall for recipes tagged "north american" and and those without the tag "north american" are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: Our model is unfair. Its recall is higher than it’s recall for recipes without the tag “North American”

To do this, we created a bar plot to verify what our false negative rates are:

<iframe src="recipe-project-main/assets/final_plot_please.html" width="800" height="600" frameborder="0"></iframe>


What this graph tells us is that non north American recipes have a higher false negative rate. However, is this difference statistically significant? We performed a permutation test to verify.

### Permutation Test Conclusion
We performed a permutation test to verify with difference of means being our test statistic and a significance level of 0.05. We used recall for our evaluation metric with our group X being the North American recipes and our group Y being the non-North American recipes. Difference of means was our test statistic of choice with a significance level of 0.05.

After performing our permutation test, we concluded with a p-value of 0.07, which was greater than our significance level of 0.05. Thus, we fail to reject our null hypothesis. This means that we cannot say our model is not fair to recipes that don't have the tags North American on them. We also cannot claim that the recall is not roughly the same for recipes with and without the tag "north american", and differences could be due to random chance.
