# Estimating ActivitySim Models with Larch

ActivitySim component models are mostly built as discrete choice models.  The
parameters for these models typically need to be estimated based on observed
survey data.  The estimation process is facilitated by the Larch package, which
is a Python package for estimating discrete choice models.  Larch is a
general-purpose package that can be used to estimate a wide variety of discrete
choice models, including the multinomial logit and nested logit models that
are commonly used in ActivitySim.

## Maximum Likelihood Estimation

The approach used to estimate the parameters of a discrete choice model is
maximum likelihood estimation (MLE).  The goal of MLE is to find the set of
parameters that maximize the likelihood of observing the choice data that we
have collected.

Finding the maximum likelihood estimates of the parameters is a non-linear
optimization problem.  To solve this problem, Larch primarily relies on the
widely-used `scipy.optimize` package, which provides a number of
[optimization algorithms](https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization)
that can be used to find the maximum likelihood estimates. Different algorithms
have different strengths and weaknesses, and the choice of algorithm can have a
significant impact on the speed and accuracy of the estimation process.  By default,
when no constraints or bounds are present, Larch uses an implementation of the
[BHHH algorithm](https://en.wikipedia.org/wiki/Berndt–Hall–Hall–Hausman_algorithm),
which is not included in scipy but is usually efficient for simple
choice models.  When constraints or bounds are present, by default Larch uses the
Larch uses the `scipy.optimize.minimize` function with the `SLSQP` algorithm when
constraints or bounds are present.  The `larch.Model.estimate` method allows the
user to specify the optimization algorithm to use via the `method` argument, which
can be set to 'BHHH', 'SLSQP', or any other algorithm supported by `scipy.optimize.minimize`.
If you are estimating a model and find the optimization is not converging as
fast as expected (or at all), you may want to try a different optimization algorithm.

## Expressing Alternative Availability

In ActivitySim, the unavailability of alternatives is typically expressed in the
utility function given in the model specification, by including a indicator variable
for unavailable alternatives, which is then attached to a large negative coefficient.
This creates a large negative utility for the unavailable alternative, which will
render it effectively unavailable in the choice model.  If *all* the alternatives
are made unavailable in this manner, this can result in a condition where no
alternative can be chosen, and ActivitySim will raise an error.

Larch, on the other hand, does typically use this approach to express alternative
availability.  Instead, Larch defines the availability of alternatives as a separate
array of values, which is not included in the utility function.  This is typically
more robust in estimation, as the computational engine can automatically shift
the utility values to avoid numerical underflow or overflow issues that can arise
when some choices are very unlikely but not strictly unavailable.
