from collections import defaultdict, Counter

import hdbscan
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns
import pandas as pd
import umap

import hits.utilities
memoized_property = hits.utilities.memoized_property

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

class Clusterer:
    def __init__(self,
                 pool,
                 **options,
                ):

        self.pool = pool

        options.setdefault('outcomes_selection_method', 'above_frequency_threshold')
        options.setdefault('outcomes_selection_kwargs', {})
        options['outcomes_selection_kwargs'].setdefault('threshold', 2e-3)

        options.setdefault('guides_selection_method', 'chi_squared_multiple')
        options.setdefault('guides_selection_kwargs', {})
        options['guides_selection_kwargs'].setdefault('multiple', 2)
        options['guides_selection_kwargs'].setdefault('n', 100)

        options.setdefault('guides_method', 'HDBSCAN')
        options.setdefault('guides_kwargs', {})

        options.setdefault('outcomes_method', 'hierarchical')
        options.setdefault('outcomes_kwargs', {})

        options['guides_kwargs'].setdefault('metric', 'cosine')
        options['outcomes_kwargs'].setdefault('metric', 'correlation')

        self.options = options

    @memoized_property
    def outcomes(self):
        if self.options['outcomes_selection_method'] == 'above_frequency_threshold':
            threshold = self.options['outcomes_selection_kwargs']['threshold']
            outcomes = self.pool.outcomes_above_simple_threshold(threshold)
        elif pd.api.types.is_list_like(self.options['outcomes_selection_method']):
            outcomes = self.options['outcomes_selection_method']
        else:
            outcomes = self.pool.canonical_outcomes

        return outcomes

    @memoized_property
    def guides(self):
        if pd.api.types.is_list_like(self.options['guides_selection_method']):
            guides = self.options['guides_selection_method']
        elif self.options['guides_selection_method'] == 'chi_squared_multiple':
            multiple = self.options['guides_selection_kwargs']['multiple']
            guides = self.pool.active_guides_above_multiple_of_max_nt(self.outcomes, multiple)
        elif self.options['guides_selection_method'] == 'chi_squared_top_n':
            n = self.options['guides_selection_kwargs']['n']
            guides = self.pool.top_n_active_guides(self.outcomes, n)
        else:
            guides = self.pool.canonical_active_guides

        return guides

    @memoized_property
    def log2_fold_changes(self):
        return self.pool.log2_fold_changes.loc[self.outcomes, self.guides]

    def perform_clustering(self, axis):
        method = self.options[f'{axis}_method']
        if method == 'hierarchical':
            func = hierarchcical
        elif method == 'HDBSCAN':
            func = HDBSCAN
        else:
            raise ValueError(method)
        
        results = func(self.log2_fold_changes, axis, **self.options[f'{axis}_kwargs'])
        assign_palette(results, axis)

        return results

    @memoized_property
    def outcome_clustering(self): 
        return self.perform_clustering('outcomes')

    @memoized_property
    def guide_clustering(self): 
        return self.perform_clustering('guides')

    @memoized_property
    def clustered_log2_fold_changes(self):
        return self.log2_fold_changes.loc[self.clustered_outcomes, self.clustered_guides]

    @memoized_property
    def clustered_guides(self):
        return self.guide_clustering['clustered_order']

    @memoized_property
    def clustered_outcomes(self):
        return self.outcome_clustering['clustered_order']

    @memoized_property
    def guide_embedding(self):
        reducer = umap.UMAP(random_state=1, metric=self.options['guides_kwargs']['metric'], n_neighbors=10, min_dist=0.2)
        embedding = reducer.fit_transform(self.log2_fold_changes.T)
        embedding = pd.DataFrame(embedding,
                                 columns=['x', 'y'],
                                 index=self.log2_fold_changes.columns,
                                )
        embedding['color'] = pd.Series(self.guide_clustering['colors'])
        embedding['gene'] = [self.pool.variable_guide_library.guide_to_gene[guide] for guide in embedding.index]
        return embedding

    @memoized_property
    def outcome_embedding(self):
        reducer = umap.UMAP(random_state=1, metric=self.options['outcomes_kwargs']['metric'], n_neighbors=10, min_dist=0.2)
        embedding = reducer.fit_transform(self.log2_fold_changes)
        embedding = pd.DataFrame(embedding,
                                 columns=['x', 'y'],
                                 index=self.log2_fold_changes.index,
                                )
        embedding['color'] = pd.Series(self.outcome_clustering['colors'])
        return embedding

    def plot_guide_embedding(self):
        fig, ax = plt.subplots(figsize=(12, 12))

        ax.scatter(x='x', y='y', color='color', data=self.guide_embedding, s=100, alpha=0.8, linewidths=(0,))

        for gene, rows in self.guide_embedding.groupby('gene', sort=False):
            n_rows, _ = rows.shape
            for first_index in range(n_rows):
                first_row = rows.iloc[first_index]
                for second_index in range(first_index + 1, n_rows):
                    second_row = rows.iloc[second_index]
                    ax.plot([first_row['x'], second_row['x']], [first_row['y'], second_row['y']], color='black', alpha=0.1)

            centroid = (rows['x'].mean(), rows['y'].mean())

            if n_rows == 1:
                num_guides_string = ''
            else:
                num_guides_string = f' ({len(rows)})'
                
            ax.annotate(f'{gene}{num_guides_string}',
                        xy=centroid,
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        size=8,
                    )

        ax.axis('off')

        return fig

def hierarchcical(l2fcs,
                  axis,
                  metric='correlation',
                  method='single',
                  **fcluster_kwargs,
                 ):

    fcluster_kwargs.setdefault('criterion', 'maxclust')
    fcluster_kwargs.setdefault('t', 5)

    if axis == 'guides':
        to_cluster = l2fcs.T
    elif axis == 'outcomes':
        to_cluster = l2fcs
    else:
        raise ValueError(axis)

    labels = list(to_cluster.index.values)

    linkage = sch.linkage(to_cluster,
                          optimal_ordering=True,
                          metric=metric,
                          method=method,
                         )

    dendro = sch.dendrogram(linkage,
                            no_plot=True,
                            labels=labels,
                           )

    clustered_order = dendro['ivl']

    cluster_ids = sch.fcluster(linkage, **fcluster_kwargs)

    # Transform from original order into the order produced by dendrogram.
    # Convert from 1-based indexing to 0-based indexing for consistency with other methods.
    cluster_assignments = [cluster_ids[labels.index(label)] - 1 for label in clustered_order]

    if axis == 'guides':
        l2fcs_reordered = l2fcs.loc[:, clustered_order]
    elif axis == 'outcomes':
        l2fcs_reordered = l2fcs.loc[clustered_order, :].T
    else:
        raise ValueError(axis)

    if metric == 'correlation':
        similarities = l2fcs_reordered.corr()
    elif metric == 'cosine':
        similarities = sklearn.metrics.pairwise.cosine_similarity(l2fcs_reordered.T)
    elif metric == 'euclidean':
        similarities = 1 / (1 + ssd.squareform(ssd.pdist(l2fcs_reordered.T)))
    else:
        similarities = None

    results = {
        'linkage': linkage,
        'dendro': dendro,
        'clustered_order': clustered_order,
        'cluster_assignments': cluster_assignments,
        'similarities': similarities,
    }

    return results

def HDBSCAN(l2fcs,
            axis,
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.2,
            metric='cosine',
            cluster_selection_method='eom',
           ):

    # cosine_distances wants samples to be rows and features to be columns
    if axis == 'guides':
        to_cluster = l2fcs.T
    elif axis == 'outcomes':
        to_cluster = l2fcs
    else:
        raise ValueError(axis)

    if metric == 'cosine':
        distances = sklearn.metrics.pairwise.cosine_distances(to_cluster)
    elif metric == 'correlation':
        distances = 1 - to_cluster.T.corr()
    elif metric == 'euclidean':
        distances = ssd.squareform(ssd.pdist(to_cluster))
    else:
        distances = None

    labels = list(to_cluster.index.values)

    distances = pd.DataFrame(distances, index=labels, columns=labels)

    clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                cluster_selection_method=cluster_selection_method,
                               )
    clusterer.fit(distances)

    linkage = clusterer.single_linkage_tree_.to_numpy()
    linkage = sch.optimal_leaf_ordering(linkage, ssd.squareform(distances))
    dendro = sch.dendrogram(linkage,
                            no_plot=True,
                            labels=labels,
                           )

    clustered_order = dendro['ivl']
    cluster_ids = clusterer.labels_

    # Transform from original order into the order produced by dendrogram.
    cluster_assignments = [cluster_ids[labels.index(l)] for l in clustered_order]

    if axis == 'guides':
        l2fcs_reordered = l2fcs.loc[:, clustered_order]
    elif axis == 'outcomes':
        l2fcs_reordered = l2fcs.loc[clustered_order, :].T
    else:
        raise ValueError(axis)

    if metric == 'correlation':
        similarities = l2fcs_reordered.corr()
    elif metric == 'cosine':
        similarities = sklearn.metrics.pairwise.cosine_similarity(l2fcs_reordered.T)
    elif metric == 'euclidean':
        similarities = 1 / (1 + ssd.squareform(ssd.pdist(l2fcs_reordered.T)))
    else:
        similarities = None

    results = {
        'clustered_order': clustered_order,
        'cluster_assignments': cluster_assignments,

        'distances': distances.loc[clustered_order, clustered_order],
        'similarities': similarities,
        'linkage': linkage,
        'original_order': labels,

        'clusterer': clusterer,
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

def get_cluster_genes(results, guide_library):
    cluster_genes = defaultdict(Counter)

    for cluster_id, guide in zip(results['cluster_assignments'], results['clustered_order']):
        gene = guide_library.guide_to_gene[guide]
        cluster_genes[cluster_id][gene] += 1
    
    return cluster_genes

def assign_palette(results, axis):
    num_clusters = len(set(results['cluster_assignments']))

    if axis == 'guides':
        palette = sns.husl_palette(num_clusters)
    elif axis == 'outcomes':
        palette = sns.color_palette('colorblind', n_colors=num_clusters)

    results['palette'] = palette
    grey = matplotlib.colors.to_rgb('grey')
    cluster_colors = {i: palette[i] for i in range(num_clusters)}
    results['cluster_colors'] = cluster_colors

    results['colors'] = {key: cluster_colors.get(i, grey) for key, i in zip(results['clustered_order'], results['cluster_assignments'])}