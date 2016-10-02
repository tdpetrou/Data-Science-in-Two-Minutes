### P-value
Value determined before a test that determines the probability that the null hypothesis will be rejected. If the test statistic produces a p-value in the rejection range and the null hypothesis is indeed correct then a type I error has been committed.

### MoM vs MLE vs MAP
MoM - Method of moments is a simple ways to estimate population parameters by using the moments of the observations. You set up a system of equations and solve for the parameters


One Form of the Method
The basic idea behind this form of the method is to:
(1) Equate the first sample moment about the origin M1=1/n∑Xi=Xbar to the first theoretical moment E(X).
(2) Equate the second sample moment about the origin M2=1n∑i=1nX2i to the second theoretical moment E(X2).
(3) Continue equating sample moments about the origin, Mk, with the corresponding theoretical moments E(Xk), k = 3, 4, ... until you have as many equations as you have parameters.
(4) Solve for the parameters.

Links
https://onlinecourses.science.psu.edu/stat414/node/193

### Moment Generating Function
Method to easily find moments of a probability distribution. M(t) = . It doesn’t always exist for all probability functions, though characteristic function always exists 
Taking the nth derivative of M evaluated at 0 yields the nth moment. 
The mgf uniquely characterizes a distribution so if two mgfs are equal then the pdfs are equivalent

### Unbalanced Classes SVM
You can assign weights to each class to more heavily weigh the unbalanced class, but even without weighting SVM’s do well.


### Covariance Matrix
Cov(X, Y) = Σ ( Xi - Xbar ) ( Yi - Ybar ) / N = Σ xiyi / N

To calculate this with a feature matrix.
Step 1: set x = X - Xbar
Step 2: Cov = x * x’/n

Diagonal elements will be variance