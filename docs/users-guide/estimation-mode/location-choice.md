# Estimation of Location Choice Models

Location choice models in ActivitySim have a few common features, which
make the process of estimation slightly different from other models.

## Understanding Size Terms

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
other parameters.

## Recreating Temporary Variables

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
