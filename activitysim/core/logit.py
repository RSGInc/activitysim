# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from activitysim.core import tracing, workflow
from activitysim.core.choosing import choice_maker
from activitysim.core.configuration.logit import LogitNestSpec
from activitysim.core.exceptions import (
    InvalidTravelError,
    ModelConfigurationError,
    TableIndexError,
)

logger = logging.getLogger(__name__)

EXP_UTIL_MIN = 1e-300
EXP_UTIL_MAX = np.inf

UTIL_MIN = np.log(EXP_UTIL_MIN, dtype=np.float64)
UTIL_UNAVAILABLE = 1000.0 * (UTIL_MIN - 1.0)


PROB_MIN = 0.0
PROB_MAX = 1.0


@dataclass
class AltsContext:
    """Representation of the alternatives without carrying around that full array."""

    min_alt_id: int
    max_alt_id: int

    def __post_init__(self):
        # e.g. for zero based zones max_alt_id = n_alts - 1
        # but for 1 based zones, we don't need to add extra padding
        self.n_rands_to_sample = max(self.max_alt_id, self.n_alts_to_cover_max_id)

    @classmethod
    def from_series(cls, ser: Union[pd.Series, pd.Index]) -> "AltsContext":
        min_alt_id = ser.min()
        max_alt_id = ser.max()
        return cls(min_alt_id, max_alt_id)

    @classmethod
    def from_num_alts(cls, num_alts: int, zero_based: bool = True) -> "AltsContext":
        if zero_based:
            offset = -1
        else:
            offset = 0
        return cls(min_alt_id=1 + offset, max_alt_id=num_alts + offset)

    @property
    def n_alts_to_cover_max_id(self) -> int:
        """If zones were non-consecutive, this could be a big over-estimate."""
        return self.max_alt_id + 1


def report_bad_choices(
    state: workflow.State,
    bad_row_map,
    df,
    trace_label,
    msg,
    trace_choosers=None,
    raise_error=True,
):
    """

    Parameters
    ----------
    bad_row_map
    df : pandas.DataFrame
        utils or probs dataframe
    msg : str
        message describing the type of bad choice that necessitates error being thrown
    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        because we  can't deduce hh_id from the interaction_dataset which is indexed on index
        values from alternatives df

    Returns
    -------
    raises RuntimeError
    """
    MAX_DUMP = 1000
    MAX_PRINT = 10

    msg_with_count = "%s %s for %s of %s rows" % (
        trace_label,
        msg,
        bad_row_map.sum(),
        len(df),
    )
    logger.warning(msg_with_count)

    df = df[bad_row_map]
    if trace_choosers is None:
        hh_ids, trace_col = tracing.trace_id_for_chooser(df.index, df)
    else:
        hh_ids, trace_col = tracing.trace_id_for_chooser(df.index, trace_choosers)
    df[trace_col] = hh_ids

    if trace_label:
        logger.info("dumping %s" % trace_label)
        state.tracing.write_csv(df[:MAX_DUMP], file_name=trace_label, transpose=False)

    # log the indexes of the first MAX_DUMP offending rows
    for idx in df.index[:MAX_PRINT].values:
        row_msg = "%s : %s in: %s = %s (hh_id = %s)" % (
            trace_label,
            msg,
            df.index.name,
            idx,
            df[trace_col].loc[idx],
        )

        logger.warning(row_msg)

    if raise_error:
        raise InvalidTravelError(msg_with_count)


def utils_to_logsums(utils, exponentiated=False, allow_zero_probs=False):
    """
    Convert a table of utilities to logsum series.

    Parameters
    ----------
    utils : pandas.DataFrame
        Rows should be choosers and columns should be alternatives.

    exponentiated : bool
        True if utilities have already been exponentiated

    Returns
    -------
    logsums : pandas.Series
        Will have the same index as `utils`.

    """

    # fixme - conversion to float not needed in either case?
    # utils_arr = utils.values.astype('float')
    utils_arr = utils.values
    if not exponentiated:
        utils_arr = np.exp(utils_arr)

    np.clip(utils_arr, EXP_UTIL_MIN, EXP_UTIL_MAX, out=utils_arr)

    utils_arr = np.where(utils_arr == EXP_UTIL_MIN, 0.0, utils_arr)

    with np.errstate(divide="ignore" if allow_zero_probs else "warn"):
        logsums = np.log(utils_arr.sum(axis=1))

    logsums = pd.Series(logsums, index=utils.index)

    return logsums


def validate_utils(
    state: workflow.State,
    utils,
    trace_label=None,
    allow_zero_probs=False,
    trace_choosers=None,
):
    """
    Validate utilities to ensure non-available choices are treated the same in EET and MC.
    For EET decisions, no conversion to probabilities is required because choices
    are made on the basis of comparing utilities (only differences matter).
    However, large negative utility values are used in practice to make choices
    unavailable based on probability calculations, which boils down to evaluating
    exp(utility). We here use this to define a minimum utility that corresponds
    to an unavailable choice.

    Parameters
    ----------
    utils : pandas.DataFrame
        Rows should be choosers and columns should be alternatives.

    trace_label : str, optional
        label for tracing bad utility or probability values

    allow_zero_probs : bool
        if True value rows in which all utility alts are UTIL_MIN will be set to
        UTIL_UNAVAILABLE.

    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        by report_bad_choices because it can't deduce hh_id from the interaction_dataset
        which is indexed on index values from alternatives df

    Returns
    -------
    utils : pandas.DataFrame
        utils with values that would lead to zero probability replaced by UTIL_UNAVAILABLE

    """
    trace_label = tracing.extend_trace_label(trace_label, "validate_utils")

    utils_arr = utils.values

    np.putmask(utils_arr, utils_arr <= UTIL_MIN, UTIL_UNAVAILABLE)

    arr_sum = utils_arr.sum(axis=1)

    if not allow_zero_probs:
        zero_probs = arr_sum <= utils_arr.shape[1] * UTIL_UNAVAILABLE
        if zero_probs.any():
            report_bad_choices(
                state,
                zero_probs,
                utils,
                trace_label=tracing.extend_trace_label(trace_label, "zero_prob_utils"),
                msg="all probabilities are zero",
                trace_choosers=trace_choosers,
            )

    utils = pd.DataFrame(utils_arr, columns=utils.columns, index=utils.index)

    return utils


def utils_to_probs(
    state: workflow.State,
    utils,
    trace_label=None,
    exponentiated=False,
    allow_zero_probs=False,
    trace_choosers=None,
    overflow_protection: bool = True,
    return_logsums: bool = False,
):
    """
    Convert a table of utilities to probabilities.

    Parameters
    ----------
    utils : pandas.DataFrame
        Rows should be choosers and columns should be alternatives.

    trace_label : str, optional
        label for tracing bad utility or probability values

    exponentiated : bool
        True if utilities have already been exponentiated

    allow_zero_probs : bool
        if True value rows in which all utility alts are EXP_UTIL_MIN will result
        in rows in probs to have all zero probability (and not sum to 1.0)
        This is for the benefit of calculating probabilities of nested logit nests

    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        by report_bad_choices because it can't deduce hh_id from the interaction_dataset
        which is indexed on index values from alternatives df

    overflow_protection : bool, default True
        Always shift utility values such that the maximum utility in each row is
        zero.  This constant per-row shift should not fundamentally alter the
        computed probabilities, but will ensure that an overflow does not occur
        that will create infinite or NaN values.  This will also provide effective
        protection against underflow; extremely rare probabilities will round to
        zero, but by definition they are extremely rare and losing them entirely
        should not impact the simulation in a measureable fashion, and at least one
        (and sometimes only one) alternative is guaranteed to have non-zero
        probability, as long as at least one alternative has a finite utility value.
        If utility values are certain to be well-behaved and non-extreme, enabling
        overflow_protection will have no benefit but impose a modest computational
        overhead cost.

    Returns
    -------
    probs : pandas.DataFrame
        Will have the same index and columns as `utils`.

    """
    trace_label = tracing.extend_trace_label(trace_label, "utils_to_probs")

    # fixme - conversion to float not needed in either case?
    # utils_arr = utils.values.astype('float')
    utils_arr = utils.values

    if allow_zero_probs:
        if overflow_protection:
            warnings.warn(
                "cannot set overflow_protection with allow_zero_probs", stacklevel=2
            )
            overflow_protection = utils_arr.dtype == np.float32 and utils_arr.max() > 85
            if overflow_protection:
                raise ValueError(
                    "cannot prevent expected overflow with allow_zero_probs"
                )
    else:
        overflow_protection = overflow_protection or (
            utils_arr.dtype == np.float32 and utils_arr.max() > 85
        )

    if overflow_protection:
        # exponentiated utils will overflow, downshift them
        shifts = utils_arr.max(1, keepdims=True)
        utils_arr -= shifts
    else:
        shifts = None

    if not exponentiated:
        # TODO: reduce memory usage by exponentiating in-place.
        #       but first we need to make sure the raw utilities
        #       are not needed elsewhere and overwriting won't hurt.
        # try:
        #     np.exp(utils_arr, out=utils_arr)
        # except TypeError:
        #     utils_arr = np.exp(utils_arr)
        utils_arr = np.exp(utils_arr)

    np.putmask(utils_arr, utils_arr <= EXP_UTIL_MIN, 0)

    arr_sum = utils_arr.sum(axis=1)

    if return_logsums:
        with np.errstate(divide="ignore" if allow_zero_probs else "warn"):
            logsums = np.log(arr_sum)
        if shifts is not None:
            logsums += np.squeeze(shifts, 1)
        logsums = pd.Series(logsums, index=utils.index)
    else:
        logsums = None

    if not allow_zero_probs:
        zero_probs = arr_sum == 0.0
        if zero_probs.any():
            report_bad_choices(
                state,
                zero_probs,
                utils,
                trace_label=tracing.extend_trace_label(trace_label, "zero_prob_utils"),
                msg="all probabilities are zero",
                trace_choosers=trace_choosers,
            )

    inf_utils = np.isinf(arr_sum)
    if inf_utils.any():
        report_bad_choices(
            state,
            inf_utils,
            utils,
            trace_label=tracing.extend_trace_label(trace_label, "inf_exp_utils"),
            msg="infinite exponentiated utilities",
            trace_choosers=trace_choosers,
        )

    # if allow_zero_probs, this may cause a RuntimeWarning: invalid value encountered in divide
    with np.errstate(
        invalid="ignore" if allow_zero_probs else "warn",
        divide="ignore" if allow_zero_probs else "warn",
    ):
        np.divide(utils_arr, arr_sum.reshape(len(utils_arr), 1), out=utils_arr)

    # if allow_zero_probs, this will cause EXP_UTIL_MIN util rows to have all zero probabilities
    np.putmask(utils_arr, np.isnan(utils_arr), PROB_MIN)

    np.clip(utils_arr, PROB_MIN, PROB_MAX, out=utils_arr)

    probs = pd.DataFrame(utils_arr, columns=utils.columns, index=utils.index)

    if return_logsums:
        return probs, logsums
    return probs


def add_ev1_random(
    state: workflow.State,
    df: pd.DataFrame,
    alt_info: AltsContext | None = None,
    alt_nrs_df: pd.DataFrame | None = None,
):
    """
    Add iid EV1 (Gumbel) random error terms to utilities for EET choice.

    Parameters
    ----------
    state : workflow.State
    df : pandas.DataFrame
        Utilities indexed by chooser and with alternatives as columns.

    Returns
    -------
    pandas.DataFrame
        Utilities with EV1 errors added.
    """
    nest_utils_for_choice = df.copy()
    assert (alt_info is None) == (
        alt_nrs_df is None
    ), "alt_info and alt_nrs_df must both be provided or omitted together"

    if alt_info is None:
        # Fallback behaviour for models where alt_info/alt_nrs_df are not provided (e.g. non-integer alts)
        rands = state.get_rn_generator().gumbel_for_df(
            nest_utils_for_choice, n=nest_utils_for_choice.shape[1]
        )
        nest_utils_for_choice += rands
        return nest_utils_for_choice

    idx_array = alt_nrs_df.values
    mask = idx_array == -999
    safe_idx = np.where(mask, 1, idx_array)  # replace -999 with a temp value inbounds
    # generate random number for all alts - this is wasteful, but ensures that the same zone
    #  gets the same random number if the sampled choice set changes between base and project
    # (alternatively, one could seed a channel for (persons x zones) and use the zone seed to ensure consistency.
    # Trade off is needing to seed (persons x zones) rows and multiindex channels to
    # avoid extra random numbers generated here. Quick benchmark suggests seeding per row is likely slower
    rands_dense = state.get_rn_generator().gumbel_for_df(
        nest_utils_for_choice, n=alt_info.n_alts_to_cover_max_id
    )
    # generate n=alt_info.max_alt_id+1 rather than n_alts so that indexing works
    # (this is drawing a random number for a redundant zeroth zone in 1 based zoning systems)
    # TODO deal with non 0->n-1 indexed land use more efficiently? ideally do where alt_nrs_df is constructed,
    #  not on the fly here. Potentially via state.get_injectable('network_los').get_skim_dict('taz').zone_ids
    rands = np.take_along_axis(rands_dense, safe_idx, axis=1)
    rands[
        mask
    ] = 0  # zero out the masked zones so they don't have the util adjustment of alt 0

    nest_utils_for_choice += rands
    return nest_utils_for_choice


def choose_from_tree(
    nest_utils, all_alternatives, logit_nest_groups, nest_alternatives_by_name
):
    for level, nest_names in logit_nest_groups.items():
        if level == 1:
            next_level_alts = nest_alternatives_by_name[nest_names[0]]
            continue
        choice_this_level = nest_utils[nest_utils.index.isin(next_level_alts)].idxmax()
        if choice_this_level in all_alternatives:
            return choice_this_level
        next_level_alts = nest_alternatives_by_name[choice_this_level]
    raise ValueError("This should never happen - no alternative found")


def make_choices_explicit_error_term_nl(
    state,
    nested_utilities,
    alt_order_array,
    nest_spec,
    trace_label,
    trace_choosers=None,
    allow_bad_utils=False,
    alts_context: AltsContext | None = None,
    alt_nrs_df: pd.DataFrame | None = None,
):
    """
    Walk down the nesting tree and make a choice at each level using EET.

    Parameters
    ----------
    state : workflow.State
    nested_utilities : pandas.DataFrame
        Utilities for nest and leaf nodes.
    alt_order_array : numpy.ndarray
        Leaf alternatives in the original ordering.
    nest_spec : dict or LogitNestSpec
        Nest specification for the choice model.
    trace_label : str
        Trace label for logging and tracing.

    Returns
    -------
    pandas.Series
        Choice indices aligned to `alt_order_array`.
    """
    if trace_label:
        state.tracing.trace_df(
            nested_utilities, tracing.extend_trace_label(trace_label, "nested_utils")
        )
    nest_utils_for_choice = add_ev1_random(
        state, nested_utilities, alts_context, alt_nrs_df
    )

    all_alternatives = set(nest.name for nest in each_nest(nest_spec, type="leaf"))
    logit_nest_groups = group_nest_names_by_level(nest_spec)
    nest_alternatives_by_name = {n.name: n.alternatives for n in each_nest(nest_spec)}

    # Apply is slow. It could *maybe* be sped up by using the fact that the nesting structure is the same for all rows:
    # Add ev1(0,1) to all entries (as is currently being done). Then, at each level, pick the maximum of the available
    # composite alternatives and set the corresponding entry to 1 for each row, set all other alternatives at this level
    # to zero. Once the tree is walked (all alternatives have been processed), take the product of the alternatives in
    # each leaf's alternative list. Then pick the only alternative with entry 1, all others must be 0.
    choices = nest_utils_for_choice.apply(
        lambda x: choose_from_tree(
            x, all_alternatives, logit_nest_groups, nest_alternatives_by_name
        ),
        axis=1,
    )
    missing_choices = choices.isnull()  # TODO: should we check for infs here too?
    if missing_choices.any() and not allow_bad_utils:
        report_bad_choices(
            state,
            missing_choices,
            nested_utilities,
            trace_label=tracing.extend_trace_label(trace_label, "bad_utils"),
            msg="no alternative selected",
            # raise_error=False,
            trace_choosers=trace_choosers,
        )
    choices = pd.Series(choices, index=nest_utils_for_choice.index)

    # In order for choice indexing to be consistent with MNL and cumsum MC choices, we need to index in the order
    #  alternatives were originally created before adding nest nodes that are not elemental alternatives
    choices = choices.map({v: k for k, v in enumerate(alt_order_array)})

    return choices


def make_choices_explicit_error_term_mnl(
    state,
    utilities,
    trace_label,
    trace_choosers=None,
    allow_bad_utils=False,
    alts_context: AltsContext | None = None,
    alt_nrs_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Make EET choices for a multinomial logit model by adding EV1 errors.

    Parameters
    ----------
    state : workflow.State
    utilities : pandas.DataFrame
        Utilities with choosers as rows and alternatives as columns.
    trace_label : str
        Trace label for logging and tracing.

    Returns
    -------
    pandas.Series
        Choice indices aligned to the utilities columns order.
    """
    if trace_label:
        state.tracing.trace_df(
            utilities, tracing.extend_trace_label(trace_label, "utilities")
        )
    utilities_incl_unobs = add_ev1_random(state, utilities, alts_context, alt_nrs_df)
    if trace_label:
        state.tracing.trace_df(
            utilities_incl_unobs,
            tracing.extend_trace_label(trace_label, "utilities_eet"),
        )
    choices = np.argmax(utilities_incl_unobs.to_numpy(), axis=1)
    missing_choices = np.isnan(choices)  # TODO: should we check for infs here too?
    if missing_choices.any() and not allow_bad_utils:
        report_bad_choices(
            state,
            missing_choices,
            utilities,
            trace_label=tracing.extend_trace_label(trace_label, "bad_utils"),
            msg="no alternative selected",
            # raise_error=False,
            trace_choosers=trace_choosers,
        )
    choices = pd.Series(choices, index=utilities_incl_unobs.index)
    return choices


def make_choices_explicit_error_term(
    state,
    utilities,
    alt_order_array,
    nest_spec=None,
    trace_label=None,
    trace_choosers=None,
    allow_bad_utils=False,
    alts_context: AltsContext | None = None,
    alt_nrs_df: pd.DataFrame | None = None,
) -> pd.Series:
    trace_label = tracing.extend_trace_label(trace_label, "make_choices_eet")
    if nest_spec is None:
        choices = make_choices_explicit_error_term_mnl(
            state,
            utilities,
            trace_label,
            trace_choosers,
            allow_bad_utils,
            alts_context,
            alt_nrs_df,
        )
    else:
        choices = make_choices_explicit_error_term_nl(
            state,
            utilities,
            alt_order_array,
            nest_spec,
            trace_label,
            trace_choosers,
            allow_bad_utils,
            alts_context,
            alt_nrs_df,
        )
    return choices


def make_choices_utility_based(
    state: workflow.State,
    utilities: pd.DataFrame,
    name_mapping=None,
    nest_spec=None,
    trace_label: str = None,
    trace_choosers=None,
    allow_bad_utils=False,
    alts_context: AltsContext | None = None,
    alt_nrs_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    trace_label = tracing.extend_trace_label(trace_label, "make_choices_utility_based")

    # For nested models, choices are mapped to `name_mapping` ordering inside the
    # EET helper. For MNL, choices already follow the utilities column order.
    choices = make_choices_explicit_error_term(
        state,
        utilities,
        name_mapping,
        nest_spec,
        trace_label,
        trace_choosers=trace_choosers,
        allow_bad_utils=allow_bad_utils,
        alts_context=alts_context,
        alt_nrs_df=alt_nrs_df,
    )
    # EET does not expose per-row random draws; return zeros for compatibility.
    rands = pd.Series(np.zeros_like(utilities.index.values), index=utilities.index)

    return choices, rands


def make_choices(
    state: workflow.State,
    probs: pd.DataFrame,
    trace_label: str = None,
    trace_choosers=None,
    allow_bad_probs=False,
) -> tuple[pd.Series, pd.Series]:
    """
    Make choices for each chooser from among a set of alternatives.
    Parameters
    ----------
    probs : pandas.DataFrame
        Rows for choosers and columns for the alternatives from which they
        are choosing. Values are expected to be valid probabilities across
        each row, e.g. they should sum to 1.
    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        by report_bad_choices because it can't deduce hh_id from the interaction_dataset
        which is indexed on index values from alternatives df
    Returns
    -------
    choices : pandas.Series
        Maps chooser IDs (from `probs` index) to a choice, where the choice
        is an index into the columns of `probs`.
    rands : pandas.Series
        The random numbers used to make the choices (for debugging, tracing)
    """
    trace_label = tracing.extend_trace_label(trace_label, "make_choices")

    # probs should sum to 1 across each row

    BAD_PROB_THRESHOLD = 0.001
    bad_probs = probs.sum(axis=1).sub(
        np.ones(len(probs.index))
    ).abs() > BAD_PROB_THRESHOLD * np.ones(len(probs.index))

    if bad_probs.any() and not allow_bad_probs:
        report_bad_choices(
            state,
            bad_probs,
            probs,
            trace_label=tracing.extend_trace_label(trace_label, "bad_probs"),
            msg="probabilities do not add up to 1",
            trace_choosers=trace_choosers,
        )

    rands = state.get_rn_generator().random_for_df(probs)

    choices = pd.Series(choice_maker(probs.values, rands), index=probs.index)

    rands = pd.Series(np.asanyarray(rands).flatten(), index=probs.index)

    return choices, rands


def interaction_dataset(
    state: workflow.State,
    choosers,
    alternatives,
    sample_size=None,
    alt_index_id=None,
    chooser_index_id=None,
):
    """
    Combine choosers and alternatives into one table for the purposes
    of creating interaction variables and/or sampling alternatives.

    Any duplicate column names in choosers table will be renamed with an '_chooser' suffix.

    Parameters
    ----------
    choosers : pandas.DataFrame
    alternatives : pandas.DataFrame
    sample_size : int, optional
        If sampling from alternatives for each chooser, this is
        how many to sample.

    Returns
    -------
    alts_sample : pandas.DataFrame
        Merged choosers and alternatives with data repeated either
        len(alternatives) or `sample_size` times.

    """
    if not choosers.index.is_unique:
        raise TableIndexError(
            "ERROR: choosers index is not unique, " "sample will not work correctly"
        )
    if not alternatives.index.is_unique:
        raise TableIndexError(
            "ERROR: alternatives index is not unique, " "sample will not work correctly"
        )

    numchoosers = len(choosers)
    numalts = len(alternatives)
    sample_size = sample_size or numalts

    # FIXME - is this faster or just dumb?
    alts_idx = np.arange(numalts)

    if sample_size < numalts:
        sample = state.get_rn_generator().choice_for_df(
            choosers, alts_idx, sample_size, replace=False
        )
    else:
        sample = np.tile(alts_idx, numchoosers)

    alts_sample = alternatives.take(sample).copy()

    if alt_index_id:
        # if alt_index_id column name specified, add alt index as a column to interaction dataset
        # permits identification of alternative row in the joined dataset
        alts_sample[alt_index_id] = alts_sample.index

    logger.debug(
        "interaction_dataset pre-merge choosers %s alternatives %s alts_sample %s"
        % (choosers.shape, alternatives.shape, alts_sample.shape)
    )

    # no need to do an expensive merge of alts and choosers
    # we can simply assign repeated chooser values
    for c in choosers.columns:
        c_chooser = (c + "_chooser") if c in alts_sample.columns else c
        alts_sample[c_chooser] = np.repeat(choosers[c].values, sample_size)

    # caller may want this to detect utils that make all alts for a chooser unavailable (e.g. -999)
    if chooser_index_id:
        assert chooser_index_id not in alts_sample
        alts_sample[chooser_index_id] = np.repeat(choosers.index.values, sample_size)

    logger.debug("interaction_dataset merged alts_sample %s" % (alts_sample.shape,))

    return alts_sample


class Nest:
    """
    Data for a nest-logit node or leaf

    This object is passed on yield when iterate over nest nodes (branch or leaf)
    The nested logit design is stored in a yaml file as a tree of dict objects,
    but using an object to pass the nest data makes the code a little more readable

    An example nest specification is in the example tour mode choice model
    yaml configuration file - example/configs/tour_mode_choice.yaml.
    """

    def __init__(self, name=None, level=0):
        self.name = name
        self.level = level
        self.product_of_coefficients = 1
        self.ancestors = []
        self.alternatives = None
        self.coefficient = 0

    def print(self):
        print(
            "Nest name: %s level: %s coefficient: %s product_of_coefficients: %s ancestors: %s"
            % (
                self.name,
                self.level,
                self.coefficient,
                self.product_of_coefficients,
                self.ancestors,
            )
        )

    @property
    def is_leaf(self):
        return self.alternatives is None

    @property
    def type(self):
        return "leaf" if self.is_leaf else "node"

    @classmethod
    def nest_types(cls):
        return ["leaf", "node"]


def validate_nest_spec(nest_spec: dict | LogitNestSpec, trace_label: str):
    keys = []
    duplicates = []
    for nest in each_nest(nest_spec):
        if nest.name in keys:
            logger.error(
                f"validate_nest_spec:duplicate nest key '{nest.name}' in nest spec - {trace_label}"
            )
            duplicates.append(nest.name)

        keys.append(nest.name)
        # nest.print()

    if duplicates:
        raise ModelConfigurationError(
            f"validate_nest_spec:duplicate nest key/s '{duplicates}' in nest spec - {trace_label}"
        )


def _each_nest(spec: LogitNestSpec, parent_nest, post_order):
    """
    Iterate over each nest or leaf node in the tree (of subtree)

    This internal routine is called by each_nest, which presents a slightly higher level interface

    Parameters
    ----------
    spec : LogitNestSpec
        Nest spec dict tree (or subtree when recursing) from the model spec yaml file
    parent_nest : Nest
        nest of parent node (passed to accumulate level, ancestors, and product_of_coefficients)
    post_order : Bool
        Should we iterate over the nodes of the tree in post-order or pre-order?
        (post-order means we yield the alternatives sub-tree before current node.)

    Yields
    ------
        spec_node : LogitNestSpec
            Nest tree spec dict for this node subtree
        nest : Nest
            Nest object with info about the current node (nest or leaf)
    """
    pre_order = not post_order

    level = parent_nest.level + 1

    if isinstance(spec, LogitNestSpec):
        name = spec.name
        coefficient = spec.coefficient
        assert isinstance(
            coefficient, int | float
        ), f"Coefficient '{name}' ({coefficient}) not a number"  # forgot to eval coefficient?
        alternatives = []
        for a in spec.alternatives:
            if isinstance(a, dict):
                alternatives.append(a["name"])
            elif isinstance(a, LogitNestSpec):
                alternatives.append(a.name)
            else:
                alternatives.append(a)

        nest = Nest(name=name)
        nest.level = parent_nest.level + 1
        nest.coefficient = coefficient
        nest.product_of_coefficients = parent_nest.product_of_coefficients * coefficient
        nest.alternatives = alternatives
        nest.ancestors = parent_nest.ancestors + [name]

        if pre_order:
            yield spec, nest

        # recursively iterate the list of alternatives
        for alternative in spec.alternatives:
            for sub_node, sub_nest in _each_nest(alternative, nest, post_order):
                yield sub_node, sub_nest

        if post_order:
            yield spec, nest

    elif isinstance(spec, str):
        name = spec

        nest = Nest(name=name)
        nest.level = parent_nest.level + 1
        nest.product_of_coefficients = parent_nest.product_of_coefficients
        nest.ancestors = parent_nest.ancestors + [name]
        nest.coefficient = parent_nest.coefficient

        yield spec, nest


def each_nest(nest_spec: dict | LogitNestSpec, type=None, post_order=False):
    """
    Iterate over each nest or leaf node in the tree (of subtree)

    Parameters
    ----------
    nest_spec : dict or LogitNestSpec
        Nest tree dict from the model spec yaml file
    type : str
        Nest class type to yield
        None yields all nests
        'leaf' yields only leaf nodes
        'branch' yields only branch nodes
    post_order : Bool
        Should we iterate over the nodes of the tree in post-order or pre-order?
        (post-order means we yield the alternatives sub-tree before current node.)

    Yields
    ------
        nest : Nest
            Nest object with info about the current node (nest or leaf)
    """
    if type is not None and type not in Nest.nest_types():
        raise ModelConfigurationError(
            "Unknown nest type '%s' in call to each_nest" % type
        )

    if isinstance(nest_spec, dict):
        nest_spec = LogitNestSpec.model_validate(nest_spec)

    for _node, nest in _each_nest(nest_spec, parent_nest=Nest(), post_order=post_order):
        if type is None or (type == nest.type):
            yield nest


def count_nests(nest_spec):
    """
    count the nests in nest_spec, return 0 if nest_spec is none
    """

    def count_each_nest(spec, count):
        if isinstance(spec, dict):
            return (
                count
                + 1
                + sum([count_each_nest(alt, count) for alt in spec["alternatives"]])
            )
        else:
            assert isinstance(spec, str)
            return 1

    return count_each_nest(nest_spec, 0) if nest_spec is not None else 0


def group_nest_names_by_level(nest_spec):
    # group nests by level, returns {level: [nest.name at that level]}
    depth = np.max([x.level for x in each_nest(nest_spec)])
    nest_levels = {x: [] for x in range(1, depth + 1)}
    for n in each_nest(nest_spec):
        nest_levels[n.level].append(n.name)
    return nest_levels
