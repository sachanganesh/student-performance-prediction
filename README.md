# Student Performance Predictor #

## Preface ##

Having spent the past few months studying quite a bit about machine learning and statistical inference, I wanted a more serious and challenging task than simply working and re-working the examples that many books and blogs make use of. There's only so many times you can look at the Iris data-set and be surprised. I wanted to work on something that was completely new to me in terms of the data, to see if I could start with the unknown and chart my way out with success.

I took on the [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) hosted at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.html).

The principal reason why I considered this dataset was that to be able to predict a student's probability of suceeding or not succeeding in a course could be invaluable in terms of personalizing education. Although the dataset is relatively small, I wanted to at least try to come away with some inherent understanding of what practical machine learning and data science looks like, and how to become better at it.

I read the [research paper](http://www3.dsi.uminho.pt/pcortez/student.pdf) that accompanied the dataset, which revealed much about the objectives of the original researchers who studied the dataset.

## Objective ##

My objective was to build a model that would predict whether or not a student would fail the math course that was being tracked. I focused on failure rates as I believed that metric to be more valuable in terms of flagging struggling students who may need more help.

To be able to preemptively assess which students may need the most attention is, in my opinion, an important step to personalized education.

To clarify, this is my amateur attempt at machine learning. I would love feedback and advice on how to improve my models or approaches.

## Data Manipulated ##

I did not consider features G1 and G2. These features track the periodic grades of each student, which provide higher insight into the final grades. I wanted to at first try to build a model that would not need these two features since it seemed like more of a challenge.

The feature G3 is the target prediction. It originally had a range of values from 0 to 20, which I mapped into a binary space. If the target value was >= 10 then it was passing (mapped to a 1), and otherwise it would be failing (mapped to a 0).

## Development Process and Results ##

I initially considered a Gaussian Naive Bayes Classifier since I've found it to be a very useful classifier across the board. I also noted that the data features appeared to be independent, which was good for the classifier since it assumes that inherently. I originally had an accuracy percentage ~1% below the paper's reported accuracy of 67.1%. After experimenting with many other models, I found that the Gaussian Naive Bayes Classifier had the most potential.

One of the quirks of the data that I noticed was that the accuracy would change depending on how I split the data for training and testing. Originally I had split the data for training to be 70% of the original data and 30% for the testing data. When I split the training and testing data 50/50, my assumption was that, with less training data available, the model would be too general, and the accuracy would decrease. For some unfathomable reason, the accuracy increased.

I then considered the false positive ratio of the results, which, in my approach, are the students that would eventually fail the class but were predicted to pass. This was a ratio I wanted to minimize.

I considered "boosting" the model by manually making it more biased towards identifying students who would fail. One of the benefits of the Naive Bayes Classifier is that it is an "on-line" classifier, or one that can partially train using new data. I iteratively trained the model on failure data multiple times to achieve a final model with an accuracy of 72% and the lowest false positive ratio of 0.42.

When I included the G1 and G2 factors, the accuracy of the boosted model improved, which was foreseeable given the correlation between periodic grade standings and final grades. However, after including these features, the boosted model performed worse than the classic model (i.e. trained without biased iteration).

My assumption is that having both models available would provide a great amount of insight; as periodic grades are revealed, the boosted model can be relied upon less and instead one can favor the classically trained model.

## Addendum ##

To clarify, this is my amateur attempt at machine learning. I would love feedback and advice on how to improve my models or approaches.

