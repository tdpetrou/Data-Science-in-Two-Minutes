# Data-Science-in-Two-Minutes
Quick descriptions and answers of common data science tasks and questions

### What is a model?
A simplistic representation of the world. Does not capture everything.

### What is the difference between reducible and irreducible error?

### What is the difference between a machine learning model and a machine learning algorithm?
Many times these words are used interchangeably but generally they have different meanings
A model (or method or technique) can be a general idea of how to interpret 
An algorithm instructs the user of the model of precisely how to execute computational steps
Examples: Linear regression is a model and least squares (or maximum likelihood) is the algorithm used to find the parameters in the model.
Neural Networks are a model with backpropagation

### Prediction vs Inference
Prediction - When given a set of inputs **X** and we are not necessarily conerned about interpretting the underlying target function *f* (could say its a black box) to predict **y**.

Inference - We care about the meaning of the predictors, their relationships, and how are they related (linear, non-linear) 




L1 vs L2
Penalizing extreme parameter values
L1 - L1 norm, diamond shaped. Easier to think of this regularization as a condition sum(abs(parameters)) < C



L1 better at sparse data. Incorrectly used on non-sparse data could yield large error.
L2 better at prediction since both highly correlated variables stay in the model.
L2 is like diversifying your portfolio. If one variable is corrupted can use other variable. L1 is more aggressive.









### Covariance Matrix
Cov(X, Y) = Σ ( Xi - Xbar ) ( Yi - Ybar ) / N = Σ xiyi / N

To calculate this with a feature matrix.
Step 1: set x = X - Xbar
Step 2: Cov = x * x’/n

Diagonal elements will be variance


### SVM vs Logistic Regression
If there is a separating hyperplane there is no guarantee logistic regression will be able to find the best one. It just guarantees the probability will be 0 or 1. This is more so for unregularized LR. SVMs might not do as well if there are random points close to the hyperplane

links
http://www.quora.com/Support-Vector-Machines/What-is-the-difference-between-Linear-SVMs-and-Logistic-Regression


 




