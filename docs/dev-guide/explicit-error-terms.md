(explicit-error-terms-dev)=
# Explicit Error Terms for Developers

Explicit Error Terms (EET) is an alternative way to simulate choices from ActivitySim's
logit models. It keeps the same systematic utilities and the same random-utility
interpretation as the standard method, but changes how the final simulated choice is
drawn.

For user-facing guidance on when to use EET, see {ref}`explicit_error_terms_ways_to_run`.

## Enabling EET

Enable EET globally in `settings.yaml`:

```yaml
use_explicit_error_terms: True
```

The top-level switch is defined in
`activitysim.core.configuration.top.SimulationSettings.use_explicit_error_terms`.
Choice simulation code reads that setting through the model compute settings and routes
supported logit simulations through the EET path.

## Default Draw Versus EET

Under the default ActivitySim simulation path, choice drawing works like this:

1. Compute systematic utilities.
2. Convert those utilities into analytical probabilities.
3. Draw one uniform random number per chooser.
4. Select the alternative whose cumulative probability interval contains that draw.

With EET enabled, the final draw step changes:

1. Compute systematic utilities.
2. Draw one iid EV1 error term for each chooser-alternative pair.
3. Add that error term to the systematic utility.
4. Choose the alternative with the highest total utility.

For multinomial logit, ActivitySim adds Gumbel draws to the utility table and takes the
row-wise maximum. For nested logit, ActivitySim applies the same idea while walking the
nest tree, preserving the configured nesting structure. For details, see
[this ATRF paper](https://australasiantransportresearchforum.org.au/frozen-randomness-at-the-individual-utility-level/).

The model being simulated does not change. EET changes how the random utility model is
sampled, not the underlying utility specification.

## Practical Effects

### Comparisons and Simulation Noise

For EET to reduce simulation noise, it is important that alternatives of a choice situation
keep the same unobserved error term in different scenario runs. This is intimately tied
to how random numbers are generated; see {ref}`random_in_detail` for the underlying
random-number stream design and the `activitysim.core.random` API.
Because unchanged alternatives can keep the same unobserved draws, changes to choices in
can only happen when the observed utility of an alternative increases. This is not the case
for the Monte Carlo simulation method, where the draws are based on probabilities, which
necessarily change for all alternatives if any observed utility changes.

This also means that one should use the same setting in all runs. Comparing a baseline run
with EET to a scenario run without EET mixes two simulation methods and makes differences
harder to interpret.

Aggregate choice patterns should remain statistically the same as for the default
probability-based method. The project test suite includes parity tests for MNL, NL,
and interaction-based simulations.

### Numerical and Debugging Behavior

EET changes the final simulation step, not the utility calculation itself. Utility
expressions, availability logic, nesting structure, and utility validation still matter in
the same way as in the default method.

In practice, EET can make some comparisons easier to interpret because the selected
alternative is the one with the highest total utility after adding the explicit error term,
rather than the one reached by a cumulative-probability threshold. That can reduce
sensitivity to small differences in the final CDF draw when comparing nearby scenarios.
It does not eliminate the need to inspect invalid or unavailable alternatives, and it does
not guarantee identical results across different RNG seeds or different model
configurations.

For shadow-priced location choice, ActivitySim resets RNG offsets between iterations when
EET is enabled so each shadow-pricing iteration uses the same sequence of random numbers.
That keeps the comparison across iterations focused on the shadow price updates instead of
changing random draws between iterations.

### Runtime

EET is slower than the default probability-based draw because it generates and processes
one random error term per chooser-alternative pair, rather than one uniform draw per
chooser after probabilities are computed. The exact runtime impact depends on the number
of alternatives, nesting structure, and interaction size. Current runtime increases are on the
order of 100% per demand model run, which is due to the non-optimized way in which location
choice is currently handled. Runtime improvement work is under way, but large improvements can
also be obtained by using Monte Carlo simulation for the sampling part of location choice, see
{ref}`explicit_error_terms_ways_to_run`.

## Implementation Details and Adding New Models

The core simulation is implemented in `activitysim.core.logit.make_choices_utility_based`. Most
calls to this function are wrapped in one of the following methods:

- `activitysim.core.simulate`
- `activitysim.core.interaction_simulate`
- `activitysim.core.interaction_sample`
- `activitysim.core.interaction_sample_simulate`

These methods have consistent implementations of EET and therefore any model using these will
automatically have EET implemented. Some models call the underlying choice simulation method
`activitysim.core.logit.make_choices` directly. For EET to work in that case, the developer has
to add a corresponding call to `logit.make_choices_utility_based`, see, e.g.,
`activitysim.abm.models.utils.cdap.household_activity_choices`. Note models that draw directly
from probability distributions, like `activitysim.abm.models.utils.cdap.extra_hh_member_choices`
do not have a corresponding EET implementation because there are no utilities to work with.


### Unavailable choices utility convention

For EET, only utility differences matter and therefore the choice between two utilities that are
very small, say -10000 and -10001, are identical to a choice between 0 and 1. For MC, utilities
have to be exponentiated and therefore floating point precision dictates the smallest and largest
utility that can be used in practice. ActivitySim historically uses a utility of -999 to make
alternatives practically unavailable. To keep consistent with this behaviour, EET also treats
alternatives with utilities smaller or equal to -999 as unavailable, see
`activitysim.core.logit.validate_utils`.
