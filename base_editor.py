import functools
from collections import Counter
from hits import utilities
import knock_knock.outcome

def at_least_n_Bs(pool, n, B):
    outcomes = []
    b_rc = utilities.reverse_complement(B)
    for c, s, d in pool.outcome_counts().index.values:
        if c == 'mismatches':
            outcome = knock_knock.outcome.MismatchOutcome.from_string(d)
            if Counter(outcome.snvs.basecalls)[b_rc] >= n:
                outcomes.append((c, s, d))
    return outcomes

def exactly_n_Bs(pool, n, B):
    outcomes = []
    b_rc = utilities.reverse_complement(B)
    for c, s, d in pool.outcome_counts().index.values:
        if c == 'mismatches':
            outcome = knock_knock.outcome.MismatchOutcome.from_string(d)
            if Counter(outcome.snvs.basecalls)[b_rc] == n:
                outcomes.append((c, s, d))
    return outcomes

def at_least_n_Bs_curried(n, B):
    return functools.partial(at_least_n_Bs, n=n, B=B)

def exactly_n_Bs_curried(n, B):
    return functools.partial(exactly_n_Bs, n=n, B=B)

def get_mismatch_outcomes(pool):
    return [(c, s, d) for c, s, d in pool.outcome_counts('perfect').index.values if c == 'mismatches']