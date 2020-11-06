import numpy as np
import scipy.cluster

def cluster(pool, outcomes, guides, metric='correlation', method='single', fixed_guide='none', min_UMIs=None):
    if isinstance(outcomes, int):
        outcomes = pool.most_frequent_outcomes(fixed_guide)[:outcomes]

    l2_fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[outcomes]
    jitter = np.random.normal(0, 1e-6, l2_fcs.shape)
    l2_fcs = l2_fcs + jitter

    if isinstance(guides, int):
        phenotype_strengths = pool.chi_squared_per_guide(outcomes, fixed_guide=fixed_guide)
        if min_UMIs is not None:
            UMIs = pool.UMI_counts('perfect').loc[fixed_guide]
            enough_UMIs = UMIs[UMIs > min_UMIs].index
            phenotype_strengths = phenotype_strengths[phenotype_strengths.index.isin(enough_UMIs)]

        guides = phenotype_strengths.index[:guides]

    guide_linkage = scipy.cluster.hierarchy.linkage(l2_fcs[guides].T,
                                                    optimal_ordering=True,
                                                    metric=metric,
                                                    method=method,
                                                   )
    guide_dendro = scipy.cluster.hierarchy.dendrogram(guide_linkage,
                                                      no_plot=True,
                                                      labels=guides,
                                                     )

    guide_order = guide_dendro['ivl']
    
    outcome_linkage = scipy.cluster.hierarchy.linkage(l2_fcs[guides],
                                                      optimal_ordering=True,
                                                      metric=metric,
                                                      method=method,
                                                     )
    outcome_dendro = scipy.cluster.hierarchy.dendrogram(outcome_linkage,
                                                        no_plot=True,
                                                        labels=outcomes,
                                                       )

    outcome_order = outcome_dendro['ivl']

    l2_fcs = l2_fcs.loc[outcome_order, guide_order]

    results = {
        'guide_order': guide_order,
        'guide_correlations': l2_fcs.corr(),
        'guide_linkage': guide_linkage,
        'original_guide_order': guides,

        'outcome_order': outcome_order,
        'outcome_correlations': l2_fcs.T.corr(),
    }

    return results
