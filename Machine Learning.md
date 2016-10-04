#Machine Learning
When presented a set of inputs, a machine can learn some thing about those inputs for some purpose.

##Types of Machine Learning
Machine learning can be divided into three broad learning types  
* Supervised - All inputs correspond with an output. The machine can be trained to predict future outputs.  
* Unsupervised - Only inputs are given. Machine can learn different structures within the input data.  
* Reinforcement - After a certain set of actions performed some feedback on performance is returned which is used by the machine to learn.  

###Supervised Machine Learning Output
* Regression - Continuous real valued response.
* Classification - Each output is a particular class. **Nominal** classes have no particular natural ordering. **Ordinal** classes have a particular order (for example: Good, Average, Bad)


###Model
A simplistic representation of the world. Does not capture everything.

### SVM vs Logistic Regression
If there is a separating hyperplane there is no guarantee logistic regression will be able to find the best one. It just guarantees the probability will be 0 or 1. This is more so for unregularized LR. SVMs might not do as well if there are random points close to the hyperplane

links
http://www.quora.com/Support-Vector-Machines/What-is-the-difference-between-Linear-SVMs-and-Logistic-Regression

### Prediction vs Inference
Prediction - When given a set of inputs **X** and we are not necessarily conerned about interpretting the underlying target function *f* (could say its a black box) to predict **y**.

Inference - We care about the meaning of the predictors, their relationships, and how are they related (linear, non-linear) 


### Recurrent Neural Net
The units form a directed cycle and thus can keep an internal state where they have different gates that determine whether
Best use case: unsegmented Hand-written digits
LSTM - long-short term memory doesn’t have vanishing gradient problem
BPTT - trained through backpropagation through time

There are different gates - input gates, forget gates of previous input, output gates
Essentially the current input and the previous input are passed to different gates. Each has different weights. They are aggregated then squashed via an activation function and finally passed to an output layer where process begins again.


links
https://s3.amazonaws.com/piazza-resources/i48o74a0lqu0/i6ys94c8na8i2/RNN.pdf?AWSAccessKeyId=AKIAJKOQYKAYOBKKVTKQ&Expires=1438359044&Signature=bks5t9RHMGBKnu2X15JWE75Hcio%3D


### Convolutional Neural Nets
Neurons are tiled in such a manner to represent overlapping visual fields.
There can be pooling layers which combine outputs from previous layers.
Can be fully connected layers.
Drop out layers reduce overfitting. Individual neurons drop out with some predefined probability

Max-Pooling: After each convolutional layer, there may be a pooling layer. The pooling layer takes small rectangular blocks from the convolutional layer and subsamples it to produce a single output from that block. There are several ways to do this pooling, such as taking the average or the maximum, or a learned linear combination of the neurons in the block. Our pooling layers will always be max-pooling layers; that is, they take the maximum of the block they are pooling.

### Generalized Linear Models
Not to be confused with General Linear Models which is the name for ordinary linear regression. General Linear Models has been abbreviated GLM with Generalized Linear Models being abbreviated GLIM but the trend is to use GLM specifically for Generalized Linear Models and have no abbreviation for General Linear Models (just call them linear models, oridary linear regression, or simple linear regression).

GLMs offer more flexibility than ordinary linear regression by allowing a non-linear relationship to hold between the response and the predictors. The right hand side is still a linear combination of coefficients and covariates (**XB** in matrix notation) but the response variable **Y** is transformed by a *link* function *g* which transformed values are then assumed to have a linear relationship with the covariates.

The response variable does not have the constraint that it is continuous, normally distributed with constant variance. The classic case is a binomial (0/1) response which clearly doesn't follow linear regression assumptions. The outcome (0/1) is not directly modeled in this case, just the log-odds using the logit link function. Poisson and negative binomial regression can be used to model discrete counts. The distribution of **Y** is different than the link function. For instance, with binomial data, Y is distributed as a binomial distribution and uses the logit link. In ordinary linear regression, **Y** is normally distributed with the identity link funciton.

The response variable must still be independent and the covariates can be transformed as in linear regression.

No closed form solution. Use maximum likelihood with newton rapson or gradient descent.

### Generative vs Discriminative Models

Generative models give a way to generate data given a particular model. They model the joint probability distribution p(x,y) and use this to calculate the posterior probability p(y|x). They model the distribution of classes p(x|y). 
Can generate sample points. For example - first pick y (say a topic) and then pick x (say a word in that topic)

p(y|x) = p(x, y) / p(x) = p(x|y)p(y)/p(x)

Generative models make assumption about distribution - as for example in Naive Bayes we assume independence of all features which can over-count evidence such as the word “Hong Kong”
Better for outlier detection and non-stationary models (dynamic data where test set is different that training)
Can be interpreted as probabilistic graphical model with a more rich interpretation of the model.
Gives you a way to generate data with p(y) and p(x|y)
Generative models assumptions don’t allow it to capture all the dependencies that are possible.
Generative models can do very well if structure is applied correctly.
Can work better with less data. 
Tend not to overfit because of restrictive assumptions

Discriminative classifiers model the posterior p(y|x) directly. They model the boundary between classes. Provide classification splits though not necessarily in a probabilistic manner, so you don’t actually need the underlying distribution

Can capture dependencies better because doesn’t have strict assumptions on distribution.
No attempt to model underlying probability distribution.


Links
http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf
http://stats.stackexchange.com/questions/12421/generative-vs-discriminative
http://www.cedar.buffalo.edu/~srihari/CSE574/Discriminative-Generative.pdf


### Parametirc vs Non-Parametric Modeling
**Parametric** - The shape of the target function *f* is assumed. The most common is linear model f(X) = β0 + β1X1 + β2X2 + ... + βpXp. With model chosen parameters are estimated from historical data. If model assumptions are wrong then a poor fit could occur. Can choose more flexible models but those are prone to overfitting.

**Non-Parametric** - No explicit assumptions about *f*. Can fit a wide variety of shapes. Since problem is not reduced to estimating parameters, much more data is needed for better fit. Hyperparameters are used instead to instruct fit.

https://www.quora.com/What-is-the-difference-between-the-parametric-model-and-the-non-parametric-model


### PCA
Many uses and abuses of PCA

When there are a large number of covariates and potentially many of them are correlated with each other, PCA can greatly reduce the number of covariates and the multicollinearity between them

A principal component is the direction of the data that explains the most variance. One that captures the most spread in the data. Take a look at the images below. If were to only examine the points where the arrows touch the line, it would be clear to see that the points on the line in the left image vary greater than those on the right. The line on the right is the line that produces the maximum variance and thus would be the first principal component. 


Each line (or hyperplane) created has a direction and variance associated with it. In PCA the direction is the eigenvector and the variance is the eigenvalue. The eigenvector with the highest eigenvalue is the principal component. This line (hyperplane) also minimizes the squared distance from the points to the line. This is not to be confused with linear regression which minimizes the squared error (given an x).

The number of eigenvectors (principal components) is equivalent to the number of dimensions of the data. Each successive eigenvector is orthogonal to the previous one. Using eigenvectors transforms your data from one space to another. These new directions are more intuitive and show more information. The image below shows this transformation



We can go a step further and reduce dimensionality by choosing to keep those eigenvectors with eigenvalues above a certain threshold.


We want lots of spread between covariates - maximum variance
To get PCA
Step 1: Get covariance matrix
Step 2: get eigenvalues of covariance matrix (do Singular value decomposition)
Step 3: normalize eigenvalues to 0 - 1. These eigenvalues represent the amount of variation retained for each variable
Step 4: USV from singular value decomposition U is new space (eigenvectors), S contains eigenvalues

The first principal component has the largest variance of the combination of covariates. It finds the direction of maximum variance and projects it on a smaller subspace. Eigenvectors point in this direction and corresponding eigenvalue gives variance in that direction
Second PC is largest variance of combination of covariates that are orthogonal to first PC

PCA: PCA sidesteps the problem of M not being diagonalizable by working directly with the n×n "covariance matrix" MTM.  Because MTM is symmetric it is guaranteed to be diagonalizable.  So PCA works by finding the eigenvectors of the covariance matrix and ranking them by their respective eigenvalues.  The eigenvectors with the greatest eigenvalues are the Principal Components of the data matrix.

Now, a little bit of matrix algebra can be done to show that the Principal Components of a PCA diagonalization of the covariance matrix MTM are the same left-singular vectors that are found through SVD (i.e. the columns of matrix V) - the same as the principal components found through PCA:

When PCA is not useful:
When doing predictive modeling, you are trying to explain the variation in the response, not the variation in the features. There is no reason to believe that cramming as much of the feature variation into a single new feature will capture a large amount of the predictive power of the features as a whole

When PCA may not be useful - for example when using random forests. The splits may happen in the last features that explain the least amount of variance among the features
The first principal component is a linear combination of all your features. The fact that it explains almost all the variability just means that most of the coefficients of the variables in the first principal component are significant.
Now the classification trees you generate are a bit of a different animal too. They do binary splits on continuous variables that best separate the categories you want to classify. That is not exactly the same as finding orthogonal linear combinations of continuous variables that give the direction of greatest variance. In fact we have recently discussed a paper on CV where PCA was used for cluster analysis and the author(s) found that there are situations where best separation is found not in the 1st few principal components but rather in the last ones.


links
https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/
http://www.stats.uwo.ca/faculty/braun/ss3850/notes/sas10.pdf
http://www.councilofdata.com/algorithms/principal-component-analysis-pca-part-1/

### Latent Dirichlet Allocation
Step 1 - Choose number of topics by eyeballing, using prior info or max likelihood
Step 2 - randomly assign each word a topic
This gives each document and word to a distribution of topics
Say Doc1 is (30, 32, 38) for the three topics and 
Word1 in Doc1 is (60, 30, 10) for the three topics
Step 3 - Iterate through each word and randomly assign it to the topic given two components.

Component 1 - prevalence of topics in that specific document p(topic|document)
Component 2 - prevalence of word across topic p(word | topic)


Say word ‘ted’ represents 20% of the words in cat1 and 40% in cat2 across all documents and we want to reassign ‘ted’ in the first document. The first document is 85% cat1 and 15% cat2. So we can weigh these probabilities and come up with a 17:6 ratio of cat1:cat2 and randomly choose how to assign the word ‘ted’

This uses bayes theorem to generate the “prior” probabilities of each word (component 1).  What is the current composition of the document. This probability will add up to 1 and then updated by the likelihood - probability word is generated from that prior.
p(topic) * p(word | topic) for each topic and then randomly select from those numbers

To initiate generative process
Initially choose topics via dirichlet distribution. This assumes a prior for the topics
And then choose words from another dirichlet distribution

Dirichlet is a continuous multivariate distribution

Latent variable is one that we infer and not directly observed

links
http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/

Go through each document, and randomly assign each word in the document to one of the K topics.
Notice that this random assignment already gives you both topic representations of all the documents and word distributions of all the topics (albeit not very good ones).
So to improve on them, for each document d…
Go through each word w in d…
And for each topic t, compute two things: 1) p(topic t | document d) = the proportion of words in document d that are currently assigned to topic t, and 2) p(word w | topic t) = the proportion of assignments to topic t over all documents that come from this word w. Reassign w a new topic, where we choose topic t with probability p(topic t | document d) * p(word w | topic t) (according to our generative model, this is essentially the probability that topic t generated word w, so it makes sense that we resample the current word’s topic with this probability). (Also, I’m glossing over a couple of things here, in particular the use of priors/pseudocounts in these probabilities.)
In other words, in this step, we’re assuming that all topic assignments except for the current word in question are correct, and then updating the assignment of the current word using our model of how documents are generated.
After repeating the previous step a large number of times, you’ll eventually reach a roughly steady state where your assignments are pretty good. So use these assignments to estimate the topic mixtures of each document (by counting the proportion of words assigned to each topic within that document) and the words associated to each topic (by counting the proportion of words assigned to each topic overall).

### NMF
Brute force matrix decomposition method factoring matrix C (DxW) into A (DxT) and matrix B (TxW). AxB approximately equals C. The non-negative part is useful in applications where non-negativity is a must. It can also make it easier to inspect. Smaller dimensions make it easier to store

In topic modeling
D - number of Documents
W - number of words
T - number of topics

So matrix A can be interpreted as the mixture of topics that for each document (row)
and matrix B can be interpreted as the mixture of words in that topic.

These can be converted to probabilities to form a generative model where documents are formed

Used to discover latent features
Algorithm uses ||C - AB||^2 and iterates to minimize this
Gibbs Sampling
A way to randomly sample from a complex multivariate joint probability distribution.

Step 1: pick out random feasible values of each variable
Step 2: condition on all the random variables except one. 
Step 3: Now you can use a simple uniform random variable to get a random value using the marginal distribution.
Step 4: repeat for other random variables in the joint. Once all variables have a value you have your ‘random’ point.

There is a burn-in required to get more feasible random values. Once you have sampled enough random values you can then choose from these sampled values to get more truly random values


### Expectation Maximization
Form of soft clustering where each point can be part of multiple clusters with different probabilities. We usually assume a gaussian or multinomial distribution and want to find the optimal parameters (mean, covariance) of the distribution

Begin algorithm by picking number of clusters
Pick random (smart) gaussians (multinomial) distributions that are feasible
Go through each point and do a soft clustering - assign a probability that it arose from each gaussian P(cluster 1 | x1) and so on. Use bayes rule here and just assume equal priors - all classes are equally likely
Once it has these assignments (probabilities of being in each cluster) it readjusts the parameters of the gaussians to fit points better
Find new mean by doing a weighted average of the points. So points that are 99% in one class will have more weight than points that are 20% of that class
Calculate new variance in the same way. Find a weighted average of the squared difference between the point and the old mean.
Repeat until distributions aren’t moving
Exellent Video: https://www.youtube.com/watch?v=REypj2sy_5U
http://stats.stackexchange.com/questions/72774/numerical-example-to-understand-expectation-maximization

#Evaluation

### ROC Curve

A graphical plot that plots False positive rate vs True positive rate by varying the threshold of a binary classifier. 

False positive rate(x) vs True positive rate(y)
False positive rate - out of all the samples that were actually negative, the percentage that was actually guessed positive - FP / Real Negatives

True Positive - sensitivity - recall - out of all the samples that were actually positive how many were actually guessed positive. TP / Real Positives



For example, we want to minimize false positives on spam so we set the threshold at .95. The false positive rate might be very low (.02) but the true positive rate might also be very low (.15) since the model needs to be very sure to guess spam. If we move the threshold to .5 - FPR will increase to say and TPR might be .8

Area on the curve is a good measure

Why the diagonal line with slope of 1? 
Let’s say there is 100 observations and 20 are actually positive.
And now I am given the chance to have 70% false positive rate (thats 56 positives out of 80 negatives wrong). Meaning by random if I am given 56 positives, just by random chance I should get at least 70% TPR. Let’s say I actually guess 5 / 20 right for TP and guess 56 positive wrong out of the 80 negative. I could simply reverse my decision and get 15 right for .75 TPR and only 24 wrong for .3 FPR. This is opposite (.25 TPR and .7 FPR)


###L1 vs L2
Penalizing extreme parameter values
L1 - L1 norm, diamond shaped. Easier to think of this regularization as a condition sum(abs(parameters)) < C

L1 better at sparse data. Incorrectly used on non-sparse data could yield large error.
L2 better at prediction since both highly correlated variables stay in the model.
L2 is like diversifying your portfolio. If one variable is corrupted can use other variable. L1 is more aggressive.