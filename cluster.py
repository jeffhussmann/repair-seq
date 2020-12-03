from collections import defaultdict, Counter

import numpy as np
import scipy.cluster
import hdbscan as hdbscan_module
import sklearn.metrics.pairwise

def get_outcomes_and_guides(pool, outcomes, guides, fixed_guide='none', min_UMIs=None, only_best_promoter=True):
    if isinstance(outcomes, int):
        outcomes = pool.most_frequent_outcomes(fixed_guide)[:outcomes]

    if isinstance(guides, int):
        phenotype_strengths = pool.chi_squared_per_guide(outcomes, fixed_guide=fixed_guide, only_best_promoter=only_best_promoter)
        if min_UMIs is not None:
            UMIs = pool.UMI_counts('perfect').loc[fixed_guide]
            enough_UMIs = UMIs[UMIs > min_UMIs].index
            phenotype_strengths = phenotype_strengths[phenotype_strengths.index.isin(enough_UMIs)]

        guides = phenotype_strengths.index[:guides]

    return outcomes, guides

def hierarchcical(
    pool, outcomes, guides,
    metric='correlation',
    method='single',
    fixed_guide='none',
    min_UMIs=None,
    num_outcome_clusters=5,
    num_guide_clusters=5,
    only_best_promoter=True,
):

    outcomes, guides = get_outcomes_and_guides(pool, outcomes, guides, fixed_guide=fixed_guide, min_UMIs=min_UMIs, only_best_promoter=only_best_promoter)

    l2_fcs = pool.log2_fold_changes('perfect', fixed_guide)[fixed_guide].loc[outcomes, guides]
    jitter = np.random.normal(0, 1e-6, l2_fcs.shape)
    l2_fcs = l2_fcs + jitter

    guide_linkage = scipy.cluster.hierarchy.linkage(l2_fcs.T,
                                                    optimal_ordering=True,
                                                    metric=metric,
                                                    method=method,
                                                   )
    guide_dendro = scipy.cluster.hierarchy.dendrogram(guide_linkage,
                                                      no_plot=True,
                                                      labels=guides,
                                                     )

    clustered_guide_order = guide_dendro['ivl']
    
    outcome_linkage = scipy.cluster.hierarchy.linkage(l2_fcs,
                                                      optimal_ordering=True,
                                                      metric=metric,
                                                      method=method,
                                                     )
    outcome_dendro = scipy.cluster.hierarchy.dendrogram(outcome_linkage,
                                                        no_plot=True,
                                                        labels=outcomes,
                                                       )

    clustered_outcome_order = outcome_dendro['ivl']

    l2_fcs = l2_fcs.loc[clustered_outcome_order, clustered_guide_order]

    guide_cluster_ids = scipy.cluster.hierarchy.fcluster(guide_linkage,
                                                   criterion='maxclust',
                                                   t=num_guide_clusters,
                                                  )
    # Transform from original order into the order produced by dendrogram.
    # Convert from 1-based indexing to 0-based indexing for consistency with other methods.
    guides_list = list(guides)
    guide_cluster_assignments = [guide_cluster_ids[guides_list.index(g)] - 1 for g in clustered_guide_order]

    outcome_cluster_ids = scipy.cluster.hierarchy.fcluster(outcome_linkage,
                                                           criterion='maxclust',
                                                           t=num_outcome_clusters,
                                                          )

    # Same transformation of indices as for guides.
    outcomes_list = list(outcomes)
    outcome_cluster_assignments = [outcome_cluster_ids[outcomes_list.index(o)] - 1 for o in clustered_outcome_order]

    cluster_guides = defaultdict(list)
    cluster_genes = defaultdict(Counter)

    for cluster_id, guide in zip(guide_cluster_assignments, clustered_guide_order):
        cluster_guides[cluster_id].append(guide)
        gene = pool.variable_guide_library.guide_to_gene[guide]
        cluster_genes[cluster_id][gene] += 1

    results = {
        'clustered_guide_order': clustered_guide_order,
        'guide_cluster_assignments': guide_cluster_assignments,

        'guide_correlations': l2_fcs.corr(),
        'guide_linkage': guide_linkage,
        'original_guide_order': guides,

        'clustered_outcome_order': clustered_outcome_order,
        'outcome_cluster_assignments': outcome_cluster_assignments,

        'outcome_correlations': l2_fcs.T.corr(),
        'outcome_linkage': outcome_linkage,

        'cluster_guides': cluster_guides,
        'cluster_genes': cluster_genes,
    }

    return results

def hdbscan(
    pool, outcomes, guides,
    min_cluster_size=2,
    min_samples=1,
    cluster_selection_epsilon=0.2,
):
    outcomes, guides = get_outcomes_and_guides(pool, outcomes, guides)
    l2_fcs = pool.log2_fold_changes('perfect', 'none')['none'].loc[outcomes, guides]

    # cosine_distances wants samples to be rows and features to be columns, hence transpose
    guide_distances = sklearn.metrics.pairwise.cosine_distances(l2_fcs.T)

    guide_clusterer = hdbscan_module.HDBSCAN(metric='precomputed',
                                       min_cluster_size=min_cluster_size,
                                       min_samples=min_samples,
                                       cluster_selection_epsilon=cluster_selection_epsilon,
                                      )
    guide_clusterer.fit(guide_distances)

    guide_linkage = guide_clusterer.single_linkage_tree_.to_numpy()
    guide_linkage = scipy.cluster.hierarchy.optimal_leaf_ordering(guide_linkage, guide_distances)
    guide_dendro = scipy.cluster.hierarchy.dendrogram(guide_linkage,
                                                      no_plot=True,
                                                      labels=guides,
                                                     )

    clustered_guide_order = guide_dendro['ivl']
    guide_cluster_ids = guide_clusterer.labels_

    # Transform from original order into the order produced by dendrogram.
    guides_list = list(guides)
    guide_cluster_assignments = [guide_cluster_ids[guides_list.index(g)] for g in clustered_guide_order]

    cluster_guides = defaultdict(list)
    cluster_genes = defaultdict(Counter)

    for cluster_id, guide in zip(guide_cluster_assignments, clustered_guide_order):
        cluster_guides[cluster_id].append(guide)
        gene = pool.variable_guide_library.guide_to_gene[guide]
        cluster_genes[cluster_id][gene] += 1

    outcome_distances = sklearn.metrics.pairwise.cosine_distances(l2_fcs)
    outcome_clusterer = hdbscan_module.HDBSCAN(metric='precomputed',
                                       min_cluster_size=min_cluster_size,
                                       min_samples=min_samples,
                                       cluster_selection_epsilon=cluster_selection_epsilon,
                                      )
    outcome_clusterer.fit(outcome_distances)

    outcome_linkage = outcome_clusterer.single_linkage_tree_.to_numpy()
    outcome_linkage = scipy.cluster.hierarchy.optimal_leaf_ordering(outcome_linkage, outcome_distances)
    outcome_dendro = scipy.cluster.hierarchy.dendrogram(outcome_linkage,
                                                      no_plot=True,
                                                      labels=outcomes,
                                                     )

    clustered_outcome_order = outcome_dendro['ivl']
    outcome_cluster_ids = outcome_clusterer.labels_

    # Transform from original order into the order produced by dendrogram.
    outcomes_list = list(outcomes)
    outcome_cluster_assignments = [outcome_cluster_ids[outcomes_list.index(g)] for g in clustered_outcome_order]

    l2_fcs = l2_fcs.loc[clustered_outcome_order, clustered_guide_order]

    results = {
        'clustered_guide_order': clustered_guide_order,
        'guide_cluster_assignments': guide_cluster_assignments,

        'guide_correlations': l2_fcs.corr(),
        'guide_linkage': guide_linkage,
        'original_guide_order': guides,

        'guide_clusterer': guide_clusterer,

        'clustered_outcome_order': clustered_outcome_order,
        'outcome_cluster_assignments': outcome_cluster_assignments,

        'outcome_correlations': l2_fcs.T.corr(),
        'outcome_linkage': outcome_linkage,
        'original_outcome_order': guides,

        'outcome_clusterer': outcome_clusterer,

        'cluster_guides': cluster_guides,
        'cluster_genes': cluster_genes,
    }

    return results

def get_cluster_blocks(cluster_assignments):
    ''' tuples of inclusive index boundaries of connected blocks of cluster ids '''

    cluster_ids = set(cluster_assignments)

    cluster_blocks = {}

    for cluster_id in cluster_ids:
        idxs = [idx for idx, c_id in enumerate(cluster_assignments) if c_id == cluster_id]
        
        if len(idxs) == 1:
            blocks = [(idxs[0], idxs[0])]
        else:
            blocks = []
            current_value = idxs[0]
            block_start = current_value

            for current_index in range(1, len(idxs)):
                previous_value = current_value
                current_value = idxs[current_index]
                if current_value - previous_value == 1:
                    continue
                else:
                    block_end = idxs[current_index - 1]
                    blocks.append((block_start, block_end))
                    block_start = current_value

            block_end = idxs[current_index]
            blocks.append((block_start, block_end))

        cluster_blocks[cluster_id] = blocks
    
    return cluster_blocks