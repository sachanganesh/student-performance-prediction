# Student Performance Prediction #

## Preface ##

Having spent the past few months studying quite a bit about machine learning and statistical inference, I wanted a more serious and challenging task than simply working and re-working the examples that many books and blogs make use of. There's only so many times you can look at the Iris data-set and be surprised. I wanted to work on something that was completely new to me in terms of the data, to see if I could start with the unknown and chart my way out with success.

Dataset: [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
Accompanying Paper: [Using Data Mining to Predict Secondary School Student Performance](http://www3.dsi.uminho.pt/pcortez/student.pdf)

## Objective ##

My objective was to build a model that would predict whether or not a student would fail the math course that was being tracked. I focused on failure rates as I believed that metric to be more valuable in terms of flagging struggling students who may need more help.

To be able to preemptively assess which students may need the most attention is, in my opinion, an important step to personalized education.

## Process ##

The target value is `G3`, which, according to the accompanying paper of the dataset, can be binned into a passing or failing classification. If `G3` is greater than or equal to 10, then the student passes. Otherwise, she fails.

The data can be reduced to 4 fundamental features:
1. `G2` score
2. `G1` score
3. `School`
4. `Absences`

A support vector linear classifier was used as the model, with a regularization factor of 100.

On average, the resulting predictive capacity starts at 70% accuracy with no grade knowledge, but improves to nearly 88% accuracy with `G1` and `G2` grade knowledge.
