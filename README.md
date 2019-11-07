# Student Performance Prediction #

## Preface ##

Having spent the past few months studying quite a bit about machine learning and statistical inference, I wanted a more serious and challenging task than simply working and re-working the examples that many books and blogs make use of. After all, there's only so many times you can look at the Iris dataset and be surprised. I wanted to work on something that was completely new to me in terms of the data, to see if I could start with the unknown and chart my way out with success.

**Dataset:** [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

**Accompanying Paper:** [Using Data Mining to Predict Secondary School Student Performance](http://www3.dsi.uminho.pt/pcortez/student.pdf)

## Objective ##

My objective was to build a model that would predict whether or not a student would fail the math course that was being tracked. I focused on failure rates as I believed that metric to be more valuable in terms of flagging struggling students who may need more help.

To be able to preemptively assess which students may need the most attention is, in my opinion, an important step to personalized education.

## Process ##

The target value is `G3`, which, according to the accompanying paper of the dataset, can be binned into a passing or failing classification. If `G3` is greater than or equal to 10, then the student passes. Otherwise, she fails. Likewise, the `G1` and `G2` features are binned in the same manner.

The data can be reduced to 4 fundamental features, in order of importance:
1. `G2` score
2. `G1` score
3. `School`
4. `Absences`

When no grade knowledge is known, `School` and `Absences` capture most of the predictive basis. As grade knowledge becomes available, `G1` and `G2` scores alone are enough to achieve over 90% accuracy. I experimentally discovered that the model performs best when it uses only 2 features at a time for each experiment.

The model is a linear support vector machine with a regularization factor of 100. This model performed the best when compared to other models, such as naive bayes, logistic regression, and random forest classifiers.

## Results ##

The following results have been averaged over 5 trials.

| Features Considered 	| G1 & G2 	| G1 & School 	| School & Absences 	|
|---------------------	|:-------:	|:-----------:	|:-----------------:	|
| Paper Accuracy      	|   0.919 	|       0.838 	|             0.706 	|
| My Model Accuracy   	|  0.9165 	|      0.8285 	|            0.6847 	|
| False Pass Rate     	|   0.096 	|        0.12 	|             0.544 	|
| False Fail Rate     	|   0.074 	|      0.1481 	|            0.2185 	|

[Why these metrics?](https://github.com/sachanganesh/student-performance-prediction/issues/1#issuecomment-508577754)

## Discussion ##

Without any prior academic performance in similar courses, the problem is difficult to solve; however, my model achieves 68% accuracy using only the school the student attends and the number of absences that they accrue to judge whether or not they fail. What is interesting is that my model, with these parameters, has a false pass rate of over 50%, meaning that it classifies more than half of the students who end up failing as passing instead. This number falls drastically as more information becomes available and better parameters are used, but it highlights one major area of improvement for the model.

To achieve their performance noted above, the original authors had to alternate models for each experiment, using both support vector machines and naive bayes. My support vector machine's performance closely follows the original author's results and displays a more streamlined approach to solving the problem, as the underlying model does not change. In addition, the original authors made use of all variables (excluding grade knowledge) in achieving the stated 70.6% accuracy in the third experiment, while my model makes use of only two parameters at a time to achieve similar results.
