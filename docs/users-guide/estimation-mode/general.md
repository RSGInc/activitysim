
# Estimation Mode

ActivitySim includes the ability to re-estimate submodels using choice model estimation
tools. It is possible to output the data needed for estimation and then use more or less
any parameter estimation tool to find the best-fitting parameters for each model, but
ActivitySim has a built-in integration with the [`larch`](https://larch.driftless.xyz)
package, which is an open source Python package for estimating discrete choice models.

## Estimation Workflow, Summarized

The general workflow for estimating models is shown in the following figures and
explained in more detail below.

![estimation workflow](https://activitysim.github.io/activitysim/develop/_images/estimation_tools.jpg)

First, the user converts their household travel survey into ActivitySim-format
households, persons, tours, joint tour participants, and trip tables.  The
households and persons tables must have the same fields as the synthetic population
input tables since the surveyed households and persons will be run through the same
set of submodels as the simulated households and persons.

The ActivitySim estimation example [``scripts\infer.py``](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_estimation/scripts/infer.py)
module reads the ActivitySim-format household travel survey files and checks for
inconsistencies in the input tables versus the model design, and calculates
additional fields such as the household joint tour frequency based on the trips
and joint tour participants table.  Survey households and persons observed choices
much match the model design (i.e. a person cannot have more work tours than the model
allows).

ActivitySim is then run in estimation mode to read the ActivitySim-format
travel survey files, and apply the ActivitySim submodels to write estimation data bundles
(EDBs) that contains the model utility specifications, coefficients, chooser data,
and alternatives data for each submodel.

The relevant EDBs are read and transformed into the format required by the model
estimation tool (i.e. larch) and then the coefficients are re-estimated. The
``activitysim.estimation.larch`` library is included for integration with larch
and there is a Jupyter Notebook estimation example for most core submodels.
Certain kinds of changes to the model specification are allowed during the estimation
process, as long as the required data fields are present in the EDB.  For example,
the user can add new expressions that transform existing data, such as converting
a continuous variable into a categorical variable, a polynomial transform, or a
piecewise linear form.  More intensive changes to the model specification, such as
adding data that is not in the EDB, or adding new alternatives, are generally not
possible without re-running the estimation mode to write a new EDB.

Based on the results of the estimation, the user can then update the model
specification and coefficients file(s) for the estimated submodel.

## Estimating ActivitySim Models with Larch

ActivitySim component models are mostly built as discrete choice models.  The
parameters for these models typically need to be estimated based on observed
survey data.  The estimation process is facilitated by the Larch package, which
is a Python package for estimating discrete choice models.  Larch is a
general-purpose package that can be used to estimate a wide variety of discrete
choice models, including the multinomial logit and nested logit models that
are commonly used in ActivitySim.  This section highlights some of the features
of Larch, particularly as they relate to ActivitySim, as there are a few subtle
differences between the two packages that users should be aware of when
estimating models.

### Maximum Likelihood Estimation

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
which is not included in scipy but is usually efficient for simple, well specified
choice models without any constraints.  When constraints or bounds are present, by default Larch uses the
Larch uses the `scipy.optimize.minimize` function with the `SLSQP` algorithm.
The `larch.Model.estimate` method allows the
user to specify the optimization algorithm to use via the `method` argument, which
can be set to 'BHHH', 'SLSQP', or any other algorithm supported by `scipy.optimize.minimize`.
If you are estimating a model and find the optimization is not converging as
fast as expected (or at all), you may want to try a different optimization algorithm.

### Model Overspecification

When using ActivitySim for simulation, there are generally few limitations or
requirements on the uniqueness of data elements. For example, it may end up being
confusing for a user, but there is nothing
fundamentally wrong with having two different variables in the model specification
that both represent "income" but have different scales, or with having
alternative-specific constants for all the alternatives.
In model estimation, however, this can lead to problems.  Having two data elements
that are perfectly correlated (e.g. two different variables that both represent "income")
or having a full set of alternative-specific values for all the alternatives can lead to
numerical problems in the estimation process, as the log likelihood function will
have flat areas and will not have a unique maximum.  This is called "model overspecification".
In Larch, the user is warned if an estimated model appears to be overspecified,
see the Larch documentation [for details](https://larch.driftless.xyz/v6.0/user-guide/choice-models.html#overspecification).

### Recreating Temporary Variables

When writing out estimation data bundles, ActivitySim may omit certain
temporary variables included in a model spec.  For example, in the example
workplace location choice model, the spec creates a temporary variable
["_DIST"](https://github.com/ActivitySim/activitysim-prototype-mtc/blob/7da9d6d6deca670cc4701fea749a270ab6fe77aa/configs/workplace_location.csv#L2)
which is then reused in several subsequent expressions.  When the model's
estimation data bundle is written out, the "_DIST" variable may not be
included[^1]. This is not a problem when simply re-estimating the parameters
of the current model specification, as all of the piecewise linear transformations
that use "_DIST" are included.  However, if the user wanted to change those
piecewise linear transformations (e.g. by moving the breakpoints), the
absence of the "_DIST" value will be relevant.

[^1]: Future versions of ActivitySim may include these values in the EDB output.

If the missing temporary value can be reconstructed from the data that *is*
included in the EDB, it can be added back into the model's data.  For example,
here we reconstitute the total distance by summing up over the piecewise
component parts:

```{python}
model.data["_DIST"] = (
    model.data.util_dist_0_1
    + model.data.util_dist_1_2
    + model.data.util_dist_2_5
    + model.data.util_dist_5_15
    + model.data.util_dist_15_up
)
```

Note in this expression, we are modifying `model.data`, i.e. the data attached
to the model.  If have other raw data available in our estimation notebook,
e.g. from running `model, data = component_model(..., return_data=True)`, it
is not sufficient to manipulate `data` itself, we must manipulate `model.data`
or otherwise re-attach any data changes to the model, or else the changes will
not show up in estimation.

### Expressing Alternative Availability

In ActivitySim, the unavailability of alternatives is typically expressed in the
utility function given in the model specification, by including a indicator variable
for unavailable alternatives, which is then attached to a large negative coefficient.
This creates a large negative utility for the unavailable alternative, which will
render it effectively unavailable in the choice model.  If *all* the alternatives
are made unavailable in this manner, this can result in a condition where no
alternative can be chosen, and ActivitySim will raise an error.

When estimating models in Larch for use with ActivitySim, it is totally acceptable and
appropriate to use this approach to express alternative availability,
by embedding it in the utility function. This will greatly simplify the process
of subsequently transferring the resulting model specification and parameters
back to the ActivitySim model.  However, it is important to note that this
approach is not the only way to express alternative availability in Larch.

Larch includes a system to define the availability of alternatives explicitly as a
[separate array of values](https://larch.driftless.xyz/dev/user-guide/choice-models.html#availability),
which is not included in the utility function.  This is
more robust in estimation, as the Larch computational engine can (and will)
automatically shift the utility values to avoid numerical underflow or overflow
issues that can arise when some choices are very unlikely but not strictly unavailable.
When using the ActivitySim style of expressing alternative availability, the onus
is entirely on the user to ensure that the utility values are not so large or small
that they cause numerical problems. If this is not checked, it is possible that
the model will appear to be estimating correctly in Larch, but the resulting model
will underflow in ActivitySim, resulting in an error when the model is run.

The scripts that build Larch models from estimation data bundles
(`activitysim.estimation.larch`) will attempt to identify unavailability flags
in the utility specifications, and when such flags are found it will automatically
convert them to the Larch availability array format. However, since specification
files can be complex, and the unavailability flags can be expressed in many different
ways, it is possible that the automatic detection will not always work as expected.
It is a good idea to check the
[choice and availability summary](https://larch.driftless.xyz/dev/user-guide/analysis.html#choice-and-availability-summary)
on the Larch model to confirm that the availability of alternatives is being
processes as expected.

### Components that have Related Models

Within ActivitySim, it is possible for multiple parts of model components
to share a common set of coefficients.  It is even possible for completely
separate components to do so.  For example, in the MTC example model,
the joint tour destination choice model and the non-mandatory tour destination
choice model share a common set of coefficients written in a single file.
To re-estimate these coefficients, the user must simultaneously work with all
the estimation survey data from both models.

In Larch, the case of two models sharing coefficients is handled by creating two separate
`Model` objects, one for each model, and then using the `ModelGroup` object to link them
together.  The `ModelGroup` object allows the user to specify a set of common parameters
for two or more models, and then estimate them together. In the case of re-estimating
the joint tour destination choice model and the non-mandatory tour destination
choice model, it may be that both models have a similar (or even identical)
utility structure. In other cases, the linked models may have different utility
structures, which share a subset of parameters, but also may have other parameters
that are unique to each model.  In either case, when using the `ModelGroup` object,
parameters are identified as being linked across models by having a common name.

There are also components in ActivitySim where a single component can embed multiple
discrete choice models which share details, but each sub-model can have a different
utility structure or different sets of parameters.  For example, the `tour_mode_choice`
component has a coefficient template file, which allows the model developer to
specify a different set of coefficients for each tour purpose, which are otherwise
processed using a common utility function.  This is implemented in Larch with the
`ModelGroup` object, where each purpose is represented as a separate `Model` object,
and the `ModelGroup` object is used to link them together.  This logic is further
extended by including the at-work subtour mode choice component, which allows for
the joint estimation of the tour mode choice model and the at-work subtour mode
choice model, which very reasonably share numerous parameters, but also have a few
differences.  Similarly, the stop frequency and CDAP models are implemented for
Larch estimation as `ModelGroup` objects, segmented on tour purpose and household
size, respectively.

When estimating a `ModelGroup` object, process for estimating the likelihood
maximizing parameters is the same as for a single model: the log likelihood is computed
for each observation (i.e. chooser) in the data set according to the parameters, model,
and data for that chooser, and the overall log likelihood is the sum of all the
chooser log likelihoods.  By using this approach, the rest of the estimation process is
the same as for a single model, including finding parameter estimates, the standard
error of those estimates, and any statistical tests or interpretations that are
desired.

### Components with Size Terms

Location choice models in ActivitySim (and in discrete choice modeling
in general) usually include a "size" term.  The size term is a measure
of the quantity of the alternative, which in location choice models is
typically a geographic area that contains multiple distinct alternatives.
For example, in a workplace location choice model, the size term might
be the number of jobs in the zone. In practice, the size term is a statistical
approximation of the number of opportunities available in the zone, and
can be composed of multiple components, such as the number of employers, the
number of households, and/or the number of retail establishments.

The size term is included in the utility function of the model, but it is
expressed differently from other qualitative measures.  The typical model
specification for a location choice model will include a utility function
given in a spec file, which will represent the quality of the alternative(s)
that are being considered.  Put another way, the "regular" utility function
is a measure of the quality of the alternative, while the size term is a
measure of the quantity of the alternative.

In ActivitySim, size terms appear not in the utility spec files, but instead
are expressed in a separate "size term" spec file, typically named
"destination_choice_size_terms.csv".  This one file contains all of the size
terms for all of the location choice models.

When using Larch for model estimation, size terms can be estimated alongside the
other parameters. The `update_size_spec` function in the `activitysim.estimation.larch` library
allows the user to update the size term specification for a model.  This function
takes the existing size term specification and updates the appropriate rows
that correspond to the model being re-estimated.  The resulting updated size
term output file will also include the (unmodified) size term specification
for all other size-based models.  When copying the revised size term specification
to the model configuration, the user should be careful that re-estimation updates
from multiple models are not inadvertently overwriting each other.

## Example Notebooks

ActivitySim includes a collection of Jupyter notebooks with
interactive re-estimation examples for many core submodels, which can be found in the
GitHub repository under the [`activitysim/examples/example_estimation/notebooks`](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/example_estimation/notebooks)
directory.  Most of these notebooks demonstrate the process of re-estimating model
parameters, without changing the model specification, i.e. finding updated values
for coefficients without changing the mathematical form of a model's utility
function.

### Examples that include Re-Specification

A selection of these notebooks have also been updated to demonstrate
the process of estimating model parameters and also *changing the model specification*.
These notebooks generally include instrucations and a demonstration of how to
modify the model specification, and then re-estimate the model parameters, as
well as how to compare the results of the original and modified models side-by-side,
which can be useful for understanding the impact of the changes made, and conducting
statistical tests to determine if the changes made are statistically significant.

The following notebooks include examples of modifying the model specification:

-   `03_work_location.ipynb`: This notebook includes a demonstration of
    modification to the SPEC file for a destination choice model, using the
    "interact-sample-simulate" type model.
-   `04_auto_ownership.ipynb`: This notebook includes a demonstration of
    modification to the SPEC file for the
    auto ownership model. It shows an example of an edit in the utility function
    for a "simple simulate" type model.
-   `06_cdap.ipynb`: This notebook includes a demonstration of modification to
    the SPEC file for the CDAP model. This model has a complex structure that is
    unique among the ActivitySim component models.
