import pandas as pd

import knock_knock.target_info
from hits import utilities

def convert_to_sgRNA_coords(target_info, a):
    ''' given a in coordinates relative to anchor on + strand, convert to relative to sgRNA beginning and sgRNA strand'''
    if target_info.sgRNA_feature.strand == '+':
        return (a + target_info.anchor) - target_info.sgRNA_feature.start 
    else:
        return target_info.sgRNA_feature.end - (a + target_info.anchor)

def convert_to_anchor_coords(target_info, s):
    ''' given s in coordinates relative to sgRNA beginning and sgRNA strand, convert to anchor coords + coords'''
    if target_info.sgRNA_feature.strand == '+':
        return (s + target_info.sgRNA_feature.start) - target_info.anchor 
    else:
        return (target_info.sgRNA_feature.end - s) - target_info.anchor

def convert_deletion(d, source_target_info, dest_target_info):
    ''' deletions are defined by starts_ats and length.
    When switching between anchor/+ and sgRNA/sgRNA strand coordinate, starts_ats may become ends_ats.
    '''
    
    start_end_pairs = list(zip(d.starts_ats, d.ends_ats))
    sgRNA_coords = [(convert_to_sgRNA_coords(source_target_info, s), convert_to_sgRNA_coords(source_target_info, e)) for s, e in start_end_pairs]
    anchor_coords = [(convert_to_anchor_coords(dest_target_info, s), convert_to_anchor_coords(dest_target_info, e)) for s, e in sgRNA_coords]
    anchor_coords = [sorted(pair) for pair in anchor_coords]
    starts_ats = sorted([s for s, e in anchor_coords])
    return knock_knock.target_info.DegenerateDeletion(starts_ats, d.length)

def convert_insertion(ins, source_target_info, dest_target_info):
    ''' insertion are defined by starts_afters and seqs
    When switching between anchor/+ and sgRNA/sgRNA strand coordinate, starts_afters may become starts_before,
    and seqs maybe be reverse complemented.
    '''
    before_after_pairs = [(s, s + 1) for s in ins.starts_afters]
    sgRNA_coords = [(convert_to_sgRNA_coords(source_target_info, b), convert_to_sgRNA_coords(source_target_info, a)) for a, b in before_after_pairs]
    anchor_coords = [(convert_to_anchor_coords(dest_target_info, b), convert_to_anchor_coords(dest_target_info, a)) for a, b in sgRNA_coords]
    anchor_coords = [sorted(pair) for pair in anchor_coords]
    starts_afters = sorted([s for s, e in anchor_coords])
    if source_target_info.sgRNA_feature.strand != dest_target_info.sgRNA_feature.strand:
        seqs = [utilities.reverse_complement(seq) for seq in ins.seqs][::-1]
    else:
        seqs = ins.seqs
    return knock_knock.target_info.DegenerateInsertion(starts_afters, seqs)

def convert_outcomes(outcomes, source_target_info, dest_target_info):
    converted = []
    for c, s, d in outcomes:
        if c == 'deletion':
            d = str(convert_deletion(knock_knock.target_info.DegenerateDeletion.from_string(d), source_target_info, dest_target_info))
        elif c == 'insertion':
            d = str(convert_insertion(knock_knock.target_info.DegenerateInsertion.from_string(d), source_target_info, dest_target_info))
        elif c == 'wild type':
            if s == 'indels':
                d = 'indels'
            s = 'clean'
        
        converted.append((c, s, d))
    
    return converted

def compare_pool_to_endogenous(pool, group, guide=None, condition=None):
    ''' '''
    endogenous_outcomes = [(c, s, d) for c, s, d in group.outcomes_by_baseline_frequency if c != 'uncategorized' and s != 'far from cut' and s != 'mismatches']
    endogenous_outcomes_converted = convert_outcomes(endogenous_outcomes, pool.target_info, group.target_info)

    outcomes = sorted(set(pool.non_targeting_fractions.index.values) | set(endogenous_outcomes_converted))

    outcomes = [(c, s, d) for c, s, d in outcomes if c != 'uncategorized']

    group_fs = group.outcome_fraction_baseline_means.loc[endogenous_outcomes]
    group_fs.index = pd.MultiIndex.from_tuples(endogenous_outcomes_converted)
    group_fs = group_fs.reindex(outcomes).fillna(0)

    pool_fs = pool.non_targeting_fractions.reindex(outcomes).fillna(0)

    fs_df = pd.DataFrame({'group': group_fs, 'pool': pool_fs})
    fs_df.index.names = ('category', 'subcategory', 'details')

    # Collapse genomic insertions to one row.
    genomic_insertion_collapsed = fs_df.loc[['genomic insertion']].groupby('subcategory').sum()

    fs_df.drop('genomic insertion', inplace=True)

    for subcategory, row in genomic_insertion_collapsed.iterrows():
        fs_df.loc['genomic insertion', subcategory, 'collapsed'] = row

    if guide is not None and condition is not None:
        group_l2fcs = group.log2_fold_change_condition_means.loc[endogenous_outcomes, condition]
        group_l2fcs.index = pd.MultiIndex.from_tuples(endogenous_outcomes_converted)
        group_l2fcs = group_l2fcs.reindex(outcomes).fillna(0)

        pool_l2fcs = pool.log2_fold_changes.reindex(outcomes)[guide]

        l2fcs_df = pd.DataFrame({'group': group_l2fcs, 'pool': pool_l2fcs})
    else:
        l2fcs_df = None

    return fs_df, l2fcs_df