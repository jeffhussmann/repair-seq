import copy
from collections import defaultdict, Counter

import hdbscan
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics.pairwise
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns
import pandas as pd
import umap

import hits.utilities
import hits.visualize

import knock_knock.outcome

from . import visualize

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
    def __init__(self, **options):
        options.setdefault('outcomes_selection_method', 'above_frequency_threshold')
        options.setdefault('outcomes_selection_kwargs', {})
        options['outcomes_selection_kwargs'].setdefault('threshold', 2e-3)

        options.setdefault('guides_selection_method', 'chi_squared_multiple')
        options.setdefault('guides_selection_kwargs', {})
        options['guides_selection_kwargs'].setdefault('multiple', 2)
        options['guides_selection_kwargs'].setdefault('n', 100)

        options.setdefault('guides_method', 'HDBSCAN')
        options.setdefault('guides_kwargs', {})

        options.setdefault('guides_seed', 0)
        options.setdefault('outcomes_seed', 0)
        options.setdefault('outcomes_min_dist', 0.2)
        options.setdefault('guides_min_dist', 0.2)

        options.setdefault('outcomes_method', 'hierarchical')
        options.setdefault('outcomes_kwargs', {})

        options.setdefault('use_high_frequency_counts', False)

        options['guides_kwargs'].setdefault('metric', 'cosine')
        options['outcomes_kwargs'].setdefault('metric', 'correlation')

        self.options = options

    def perform_clustering(self, axis):
        method = self.options[f'{axis}_method']
        if method == 'hierarchical':
            func = hierarchical
        elif method == 'HDBSCAN':
            func = HDBSCAN
        elif method == 'precomputed':
            func = process_precomputed
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
    def outcomes_with_pool(self):
        return self.log2_fold_changes.index.values

    @memoized_property
    def all_log2_fold_changes(self):
        return self.guide_log2_fold_changes(self.guide_library.guides)

    @memoized_property
    def all_guide_correlations(self):
        all_l2fcs = self.all_log2_fold_changes
        all_corrs = all_l2fcs.corr().stack()

        all_corrs.index.names = ['guide_1', 'guide_2']

        all_corrs = pd.DataFrame(all_corrs)
        all_corrs.columns = ['r']

        # Make columns out of index levels.
        for k in all_corrs.index.names:
            all_corrs[k] = all_corrs.index.get_level_values(k)
            
        guide_to_gene = self.guide_library.guide_to_gene

        all_corrs['gene_1'] = guide_to_gene.loc[all_corrs['guide_1']].values
        all_corrs['gene_2'] = guide_to_gene.loc[all_corrs['guide_2']].values

        for guide in ['guide_1', 'guide_2']:
            all_corrs[f'{guide}_is_active'] = all_corrs[guide].isin(self.guides)

        all_corrs['both_active'] = all_corrs[['guide_1_is_active', 'guide_2_is_active']].all(axis=1)

        distinct_guides = all_corrs.query('guide_1 < guide_2')
        return distinct_guides.sort_values('r', ascending=False)

    @memoized_property
    def guide_embedding(self):
        reducer = umap.UMAP(random_state=self.options['guides_seed'],
                            metric=self.options['guides_kwargs']['metric'],
                            n_neighbors=10,
                            min_dist=self.options['guides_min_dist'],
                           )
        embedding = reducer.fit_transform(self.log2_fold_changes.T)
        embedding = pd.DataFrame(embedding,
                                 columns=['x', 'y'],
                                 index=self.log2_fold_changes.columns,
                                )
        embedding['color'] = pd.Series(self.guide_clustering['colors'])
        embedding['gene'] = [self.guide_to_gene[guide] for guide in embedding.index]
        embedding['cluster_assignments'] = pd.Series(self.guide_clustering['cluster_assignments'], index=self.guide_clustering['clustered_order'])

        return embedding

    @memoized_property
    def outcome_embedding(self):
        reducer = umap.UMAP(random_state=self.options['outcomes_seed'],
                            metric=self.options['outcomes_kwargs']['metric'],
                            n_neighbors=10,
                            min_dist=self.options['outcomes_min_dist'],
                           )
        embedding = reducer.fit_transform(self.log2_fold_changes)
        embedding = pd.DataFrame(embedding,
                                 columns=['x', 'y'],
                                 index=self.log2_fold_changes.index,
                                )
        embedding['color'] = pd.Series(self.outcome_clustering['colors'])
        embedding['cluster_assignment'] = pd.Series(self.outcome_clustering['cluster_assignments'], index=self.outcome_clustering['clustered_order'])

        MH_lengths = []
        deletion_lengths = []
        directionalities = []
        fractions = []

        for pn, c, s, d in embedding.index.values:
            pool = self.pn_to_pool[pn]
            ti = self.pn_to_target_info[pn]

            if c == 'deletion':
                deletion = knock_knock.outcome.DeletionOutcome.from_string(d).undo_anchor_shift(ti.anchor)
                MH_length = len(deletion.deletion.starts_ats) - 1
                deletion_length = deletion.deletion.length
                directionality = deletion.classify_directionality(ti)
            else:
                MH_length = -1
                deletion_length = -1
                directionality = 'n/a'
                
            MH_lengths.append(MH_length)
            deletion_lengths.append(deletion_length)
            directionalities.append(directionality)

            if self.options['use_high_frequency_counts']:
                fraction = pool.high_frequency_outcome_fractions.loc[(c, s, d), 'all_non_targeting']
            else:
                fraction = pool.outcome_fractions().loc[(c, s, d), ('none', 'all_non_targeting')]

            fractions.append(fraction)

        embedding['MH length'] = MH_lengths
        embedding['deletion length'] = deletion_lengths
        embedding['directionality'] = directionalities
        embedding['fraction'] = fractions
        embedding['log10_fraction'] = np.log10(embedding['fraction'])

        effector = self.target_info.effector.name

        categories = embedding.index.get_level_values('category')

        combined_categories = []

        for category, directionality in zip(categories, embedding['directionality']):
            if category != 'deletion':
                combined_category = category
            else:
                combined_category = f'{category}, {directionality}'

            combined_categories.append(combined_category)

        combined_categories = pd.Series(combined_categories, index=embedding.index)
        categories_aliased = combined_categories.map(visualize.category_aliases[effector])

        value_to_color = visualize.category_alias_colors[effector]
        colors = categories_aliased.map(value_to_color)

        embedding['combined_categories'] = combined_categories
        embedding['categories_aliased'] = categories_aliased
        embedding['category_colors'] = colors

        embedding['sgRNA'] = embedding.index.get_level_values('pool_name').map(self.pn_to_sgRNA).values

        return embedding

    def plot_guide_embedding(self,
                             figsize=(12, 12),
                             color_by='cluster',
                             label_genes=True,
                             label_alpha=1,
                             legend_location='lower right',
                             label_size=8,
                             marker_size=None,
                             circle_edge_width=0,
                             gene_line_width=1,
                            ):
        fig, ax = plt.subplots(figsize=figsize)

        data = self.guide_embedding.copy()

        if color_by == 'cluster':
            # sort so that guides assigned to clusters are drawn on top
            data['sort_by'] = data['cluster_assignments']
            data = data.sort_values(by='sort_by')
        elif color_by == 'gamma':
            min_gamma = -0.3

            values = self.guide_library.guides_df['gamma'].loc[data.index]
            cmap = visualize.gamma_cmap

            norm = matplotlib.colors.Normalize(vmin=min_gamma, vmax=0)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

            colors = [tuple(row) for row in sm.to_rgba(values)]
            data['color'] = colors
            # sort so that strongest (negative) gammas are drawn on top
            data['sort_by'] = -values
            data = data.sort_values(by='sort_by')

            ax_p = ax.get_position()

            if legend_location == 'lower right':
                x0 = ax_p.x0 + 0.65 * ax_p.width
                y0 = ax_p.y0 + 0.25 * ax_p.height
            elif legend_location == 'upper right':
                x0 = ax_p.x0 + 0.70 * ax_p.width
                y0 = ax_p.y0 + 0.95 * ax_p.height
            else:
                raise NotImplementedError

            cax = fig.add_axes([x0,
                                y0,
                                0.25 * ax_p.width,
                                0.03 * ax_p.height,
                                ])

            colorbar = plt.colorbar(mappable=sm, cax=cax, orientation='horizontal')
            ticks = [min_gamma, 0]
            tick_labels = [str(t) for t in ticks]
            tick_labels[0] = '$\leq$' + tick_labels[0]
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(tick_labels)
            colorbar.outline.set_alpha(0)
            colorbar.set_label(f'gamma', size=6)
            cax.tick_params(labelsize=6, width=0.5)

        elif color_by in self.guide_library.cell_cycle_log2_fold_changes.index:
            phase = color_by

            values = self.guide_library.cell_cycle_log2_fold_changes.loc[phase, data.index]
            cmap = visualize.cell_cycle_cmap

            norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

            colors = [tuple(row) for row in sm.to_rgba(values)]

            data['color'] = colors

            # sort by absolute value of fold changes so that most extreme
            # points are drawn on top
            data['sort_by'] = np.abs(values)
            data = data.sort_values(by='sort_by')

            ax_p = ax.get_position()

            if legend_location == 'lower right':
                x0 = ax_p.x0 + 0.65 * ax_p.width
                y0 = ax_p.y0 + 0.25 * ax_p.height
            elif legend_location == 'upper right':
                x0 = ax_p.x0 + 0.70 * ax_p.width
                y0 = ax_p.y0 + 0.95 * ax_p.height
            else:
                raise NotImplementedError

            cax = fig.add_axes([x0,
                                y0,
                                0.25 * ax_p.width,
                                0.03 * ax_p.height,
                                ])

            colorbar = plt.colorbar(mappable=sm, cax=cax, orientation='horizontal')
            ticks = [-1, 0, 1]
            tick_labels = [str(t) for t in ticks]
            tick_labels[0] = '$\leq$' + tick_labels[0]
            tick_labels[-1] = '$\geq$' + tick_labels[-1]
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(tick_labels)
            colorbar.outline.set_alpha(0)
            colorbar.set_label(f'log$_2$ fold change\nin {phase} occupancy', size=6)
            cax.tick_params(labelsize=6, width=0.5)

        if marker_size is None:
            marker_size = figsize[0] * 100 / 12

        ax.scatter(x='x',
                   y='y',
                   color='color',
                   data=data,
                   s=marker_size,
                   alpha=0.8,
                   edgecolors='black',
                   linewidths=(circle_edge_width,),
                   #marker='s',
                  )

        if label_genes == 'all':
            for guide, row in self.guide_embedding.iterrows():
                ax.annotate(row['gene'],
                            xy=(row['x'], row['y']),
                            xytext=(2, 0),
                            textcoords='offset points',
                            ha='left',
                            va='center',
                            size=label_size,
                            alpha=label_alpha,
                           )

        elif label_genes:
            for gene, rows in self.guide_embedding.groupby('gene', sort=False):
                n_rows, _ = rows.shape
                for first_index in range(n_rows):
                    first_row = rows.iloc[first_index]
                    for second_index in range(first_index + 1, n_rows):
                        second_row = rows.iloc[second_index]
                        xs = [first_row['x'], second_row['x']]
                        ys = [first_row['y'], second_row['y']]
                        if gene_line_width > 0:
                            ax.plot(xs, ys, linewidth=gene_line_width, color='black', alpha=0.1)

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
                            size=label_size,
                            alpha=label_alpha,
                        )

        x_min = data['x'].min()
        x_max = data['x'].max()
        x_range = x_max - x_min
        buffer = 0.05 * x_range
        ax.set_xlim(x_min - buffer, x_max + buffer)

        y_min = data['y'].min()
        y_max = data['y'].max()
        y_range = y_max - y_min
        buffer = 0.05 * y_range
        ax.set_ylim(y_min - buffer, y_max + buffer)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.setp(fig.axes[0].spines.values(), alpha=0.5)

        return fig, data

    def plot_outcome_embedding(self,
                               marker_size=35,
                               alpha=0.8,
                               color_by='cluster',
                               ax=None,
                               figsize=(6, 6),
                               draw_legend=True,
                               legend_location='upper left',
                               legend_text_size=12,
                               marker='o',
                              ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        data = self.outcome_embedding.copy()

        common_kwargs = dict(
            x='x',
            y='y',
            s=marker_size,
            alpha=alpha,
            linewidths=(0,),
            clip_on=False,
            color='color',
            marker=marker,
        )

        needs_categorical_legend = False
        value_to_color = None

        if color_by is None:
            data['color'] = 'grey'
            ax.scatter(data=data,
                       **common_kwargs,
                      )

        elif color_by == 'cluster':
            # To prevent privileging specific clusters, don't
            # sort by cluster assignment, but do plot assigned
            # points after unasigned ones.
            ax.scatter(data=data.query('cluster_assignment == -1'),
                       **common_kwargs,
                      )
            ax.scatter(data=data.query('cluster_assignment != -1'),
                       **common_kwargs,
                      )  
        else:
            data_query = None

            if color_by == 'sgRNA':
                colors, value_to_color = self.sgRNA_colors()
                data['color'] = colors

                # Shuffle to randomize z-order.
                data = data.sample(frac=1)

                needs_categorical_legend = True

            elif color_by == 'category':
                effector = self.target_info.effector.name

                value_to_color = visualize.category_alias_colors[effector]

                data['color'] = data['category_colors']

                # Force insertions to be drawn on top.
                data = data.sort_values(by='categories_aliased', key=lambda s: s == 'insertion')

                needs_categorical_legend = True

            elif color_by in ['MH length', 'deletion length', 'fraction', 'log10_fraction']:
                if color_by == 'MH length':
                    min_length = 0
                    max_length = 3
                    cmap = copy.copy(plt.get_cmap('viridis_r'))
                    tick_step = 1
                    label = 'nts of flanking\nmicrohomology'
                    data_query = 'category == "deletion"'
                elif color_by == 'deletion length':
                    min_length = 0
                    max_length = 30
                    cmap = copy.copy(plt.get_cmap('Purples'))
                    tick_step = 10
                    label = 'deletion length'
                    data_query = 'category == "deletion"'
                elif color_by == 'fraction':
                    min_length = 0
                    max_length = 0.1
                    cmap = copy.copy(plt.get_cmap('YlOrBr'))
                    tick_step = 0.25
                    label = 'fraction of outcomes'
                    data_query = None
                elif color_by == 'log10_fraction':
                    min_length = -3
                    max_length = -1
                    cmap = copy.copy(plt.get_cmap('YlOrBr'))
                    tick_step = 1
                    label = 'log$_{10}$ baseline\nfraction\nof outcomes\nwithin screen'
                    data_query = None
                else:
                    raise ValueError(color_by)

                if color_by == 'MH length':
                    norm = matplotlib.colors.Normalize(vmin=min_length, vmax=max_length)
                    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                    colors = sm.to_rgba(np.arange(min_length, max_length + 1))
                    cmap = matplotlib.colors.ListedColormap(colors)
                    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, max_length + 1), cmap.N) 
                    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                else:
                    cmap.set_under('white')
                    norm = matplotlib.colors.Normalize(vmin=min_length, vmax=max_length)
                    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

                vs = self.outcome_embedding[color_by]

                colors = [tuple(row) for row in sm.to_rgba(vs)]
                data['color'] = colors

                data['sort_by'] = vs
                data = data.sort_values(by='sort_by')

                ax_p = ax.get_position()

                if 'left' in legend_location:
                    x0 = ax_p.x0 + 0.45 * ax_p.width
                elif 'middle' in legend_location:
                    x0 = ax_p.x0 + 0.5 * ax_p.width
                else:
                    x0 = ax_p.x0 + 0.75 * ax_p.width

                if 'upper' in  legend_location:
                    y0 = ax_p.y0 + 0.65 * ax_p.height
                elif 'middle' in legend_location:
                    y0 = ax_p.y0 + 0.5 * ax_p.height - 0.125 * ax_p.height
                elif 'lower' in legend_location or 'bottom' in legend_location:
                    y0 = ax_p.y0 + 0.2 * ax_p.height - 0.125 * ax_p.height

                cax = fig.add_axes([x0,
                                    y0,
                                    0.03 * ax_p.width,
                                    0.25 * ax_p.height,
                                   ])
                colorbar = plt.colorbar(mappable=sm, cax=cax)

                ticks = np.arange(min_length, max_length + 1, tick_step)
                tick_labels = [str(t) for t in ticks]
                tick_labels[-1] = '$\geq$' + tick_labels[-1]
                if min_length != 0:
                    tick_labels[0] = '$\leq$' + tick_labels[0]
                colorbar.set_ticks(ticks)
                colorbar.set_ticklabels(tick_labels)
                colorbar.outline.set_alpha(0)

                cax.tick_params(labelsize=6, width=0.5, length=2)

                cax.annotate(label,
                             xy=(0, 0.5),
                             xycoords='axes fraction',
                             xytext=(-5, 0),
                             textcoords='offset points',
                             va='center',
                             ha='right',
                             size=legend_text_size,
                            )

            else:
                # color_by is be a guide name. This might be a guide that was among
                # those used for clustering or might not.
                guide = color_by

                if guide in self.log2_fold_changes:
                    values = self.log2_fold_changes[guide]
                else:
                    values = self.guide_log2_fold_changes(guide)

                values = values.loc[data.index]
                norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
                cmap = visualize.fold_changes_cmap
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

                colors = [tuple(row) for row in sm.to_rgba(values)]
                data['color'] = colors

                # sort by absolute value of fold changes so that most extreme
                # points are drawn on top
                data['sort_by'] = np.abs(values)
                data = data.sort_values(by='sort_by')

                ax_p = ax.get_position()

                if 'left' in legend_location:
                    x0 = ax_p.x0 + 0.1 * ax_p.width
                elif 'middle' in legend_location:
                    x0 = ax_p.x0 + 0.4 * ax_p.width
                else:
                    x0 = ax_p.x0 + 0.6 * ax_p.width

                if 'upper' in  legend_location:
                    y0 = ax_p.y0 + 0.75 * ax_p.height
                elif 'middle' in legend_location:
                    y0 = ax_p.y0 + 0.5 * ax_p.height - 0.125 * ax_p.height
                elif 'lower' in legend_location or 'bottom' in legend_location:
                    y0 = ax_p.y0 + 0.3 * ax_p.height - 0.125 * ax_p.height

                cax = fig.add_axes([x0,
                                    y0,
                                    0.3 * ax_p.width,
                                    0.03 * ax_p.height,
                                   ])
                colorbar = plt.colorbar(mappable=sm, cax=cax, orientation='horizontal')
                ticks = [-2, 0, 2]
                tick_labels = [str(t) for t in ticks]
                tick_labels[0] = '$\leq$' + tick_labels[0]
                tick_labels[-1] = '$\geq$' + tick_labels[-1]
                colorbar.set_ticks(ticks)
                colorbar.set_ticklabels(tick_labels)
                colorbar.outline.set_alpha(0)
                colorbar.set_label('log$_2$ fold change\nfrom non-targeting', size=legend_text_size)
                cax.tick_params(labelsize=6, width=0.5, length=2)

                cax.annotate(guide,
                             xy=(0.5, 1),
                             xycoords='axes fraction',
                             xytext=(0, legend_text_size / 2),
                             textcoords='offset points',
                             va='bottom',
                             ha='center',
                             size=legend_text_size,
                             color=self.guide_embedding['color'].get(guide, 'black')
                            )

            if data_query is not None:
                to_plot = data.query(data_query)
            else:
                to_plot = data

            ax.scatter(data=to_plot,
                       **common_kwargs,
                      )

        x_min = data['x'].min()
        x_max = data['x'].max()
        x_range = x_max - x_min
        buffer = 0.05 * x_range
        ax.set_xlim(x_min - buffer, x_max + buffer)

        y_min = data['y'].min()
        y_max = data['y'].max()
        y_range = y_max - y_min
        buffer = 0.05 * y_range
        ax.set_ylim(y_min - buffer, y_max + buffer)

        ax.set_xticks([])
        ax.set_yticks([])

        if draw_legend and needs_categorical_legend:
            hits.visualize.draw_categorical_legend(value_to_color, ax, font_size=legend_text_size, legend_location=legend_location)

        plt.setp(fig.axes[0].spines.values(), alpha=0.5)

        return fig, data, value_to_color

def hierarchical(l2fcs,
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

def process_precomputed(l2fcs,
                        axis,
                        metric='cosine',
                        **fcluster_kwargs,
                ):

    if axis == 'guides':
        to_cluster = l2fcs.T
    elif axis == 'outcomes':
        to_cluster = l2fcs
    else:
        raise ValueError(axis)

    if axis == 'guides':
        l2fcs_reordered = l2fcs
    elif axis == 'outcomes':
        l2fcs_reordered = l2fcs.T

    labels = list(to_cluster.index.values)

    if metric == 'correlation':
        similarities = l2fcs_reordered.corr()
    elif metric == 'cosine':
        similarities = sklearn.metrics.pairwise.cosine_similarity(l2fcs_reordered.T)
    elif metric == 'euclidean':
        similarities = 1 / (1 + ssd.squareform(ssd.pdist(l2fcs_reordered.T)))
    else:
        similarities = None

    results = {
        'clustered_order': labels,
        'similarities': similarities,
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

class SinglePoolClusterer(Clusterer):
    def __init__(self, pool, **options):
        self.pool = pool
        self.guide_library = self.pool.variable_guide_library
        self.pn_to_target_info = {self.pool.short_name: self.pool.target_info}
        self.pn_to_sgRNA = {self.pool.short_name: self.pool.target_info.sgRNA}
        self.pn_to_pool = {self.pool.short_name: self.pool}
        super().__init__(**options)

    @memoized_property
    def guide_to_gene(self):
        return self.pool.variable_guide_library.guide_to_gene

    @property
    def target_info(self):
        return self.pool.target_info

    @memoized_property
    def outcomes(self):
        selection_method = self.options['outcomes_selection_method']

        if pd.api.types.is_list_like(selection_method):
            outcomes = selection_method

        elif selection_method == 'category':
            category = self.options['outcomes_selection_kwargs']['category']
            min_f = self.options['outcomes_selection_kwargs']['threshold']
            outcomes = [(c, s, d) for (c, s, d), f in self.pool.non_targeting_fractions.items() if c == category and f >= min_f]

        elif selection_method == 'above_frequency_threshold':
            threshold = self.options['outcomes_selection_kwargs']['threshold']
            outcomes = self.pool.outcomes_above_simple_threshold(frequency_threshold=threshold,
                                                                 use_high_frequency_counts=self.options['use_high_frequency_counts'],
                                                                )

        elif selection_method == 'top_n':
            num_outcomes = self.options['outcomes_selection_kwargs']['n']
            outcomes = self.pool.most_frequent_outcomes(use_high_frequency_counts=self.options['use_high_frequency_counts'])[:num_outcomes]

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
            guides = self.pool.top_n_active_guides(self.outcomes, n,
                                                   use_high_frequency_counts=self.options['use_high_frequency_counts'],
                                                  )

        else:
            guides = self.pool.canonical_active_guides

        return guides

    @memoized_property
    def log2_fold_changes(self):
        if self.options['use_high_frequency_counts']:
            l2fcs = self.pool.high_frequency_log2_fold_changes.loc[self.outcomes, self.guides]
        else:
            l2fcs = self.pool.log2_fold_changes().loc[self.outcomes, self.guides]

        index = pd.MultiIndex.from_tuples([(self.pool.short_name, *vs) for vs in l2fcs.index])
        index.names = ['pool_name'] + l2fcs.index.names
        l2fcs.index = index
        return l2fcs

    def guide_log2_fold_changes(self, guide):
        if self.options['use_high_frequency_counts']:
            l2fcs = self.pool.high_frequency_log2_fold_changes.loc[self.outcomes, guide]
        else:
            l2fcs = self.pool.log2_fold_changes().loc[self.outcomes, guide]

        index = pd.MultiIndex.from_tuples([(self.pool.short_name, *vs) for vs in l2fcs.index])
        index.names = ['pool_name'] + l2fcs.index.names
        l2fcs.index = index
        return l2fcs

class MultiplePoolClusterer(Clusterer):
    def __init__(self, pools, **options):
        self.pools = pools

        # Note: assumes all pools have the same guide library.
        self.guide_library = self.pools[0].variable_guide_library

        self.pn_to_pool = {pool.short_name: pool for pool in self.pools}
        self.pn_to_target_info = {pool.short_name: pool.target_info for pool in self.pools}
        self.pn_to_sgRNA = {pool.short_name: pool.target_info.sgRNA for pool in self.pools}
        super().__init__(**options)

    def sgRNA_colors(self):
        outcome_colors, sgRNA_to_color = hits.visualize.assign_categorical_colors(self.outcome_embedding['sgRNA'])
        return outcome_colors, sgRNA_to_color

    @memoized_property
    def common_guides(self):
        return set.intersection(*[set(pool.variable_guide_library.guides) for pool in self.pools])

    @property
    def target_info(self):
        return self.pools[0].target_info

    @memoized_property
    def guide_to_gene(self):
        guide_to_gene = {}

        for pool in self.pools:
            guide_to_gene.update(pool.variable_guide_library.guide_to_gene)

        return guide_to_gene

    @memoized_property
    def guides(self):
        if pd.api.types.is_list_like(self.options['guides_selection_method']):
            guides = self.options['guides_selection_method']

        else:
            active_guide_lists = defaultdict(list)

            for pool in self.pools:
                for i, guide in enumerate(pool.canonical_active_guides(use_high_frequency_counts=self.options['use_high_frequency_counts'])):
                    active_guide_lists[guide].append((pool.short_name, i))

            # Include guides that were active in at least two screens.
            # Only include up to 3 guides per gene, prioritized by average activity rank.

            gene_guides = defaultdict(list)

            for guide, active_pns in active_guide_lists.items():
                if len(active_pns) >= min(len(self.pools), 2):
                    average_rank = np.mean([rank for pn, rank in active_pns])
                    gene_guides[self.guide_to_gene[guide]].append((average_rank, guide))
                    
            filtered_guides = []
            for gene, guides in gene_guides.items():
                guides = [g for r, g in sorted(guides)]
                filtered_guides.extend(guides[:3])

            guides = filtered_guides

        return guides

    @memoized_property
    def pool_specific_outcomes(self):
        selection_method = self.options['outcomes_selection_method']

        pool_specific_outcomes = {}

        for pool in self.pools:
            if selection_method == 'category':
                category = self.options['outcomes_selection_kwargs']['category']
                min_f = self.options['outcomes_selection_kwargs']['threshold']
                outcomes = [(c, s, d) for (c, s, d), f in pool.non_targeting_fractions.items() if c == category and f >= min_f]

            elif selection_method == 'above_frequency_threshold':
                threshold = self.options['outcomes_selection_kwargs']['threshold']
                outcomes = pool.outcomes_above_simple_threshold(frequency_threshold=threshold,
                                                                use_high_frequency_counts=self.options['use_high_frequency_counts'],
                                                               )

            elif selection_method == 'top_n':
                num_outcomes = self.options['outcomes_selection_kwargs']['n']
                outcomes = pool.most_frequent_outcomes('none')[:num_outcomes]

            else:
                outcomes = pool.canonical_outcomes

            pool_specific_outcomes[pool.short_name] = outcomes

        return pool_specific_outcomes

    @memoized_property
    def outcomes(self):
        return [(c, s, d) for pn, c, s, d in self.outcomes_with_pool]
    
    @memoized_property
    def log2_fold_changes(self):
        all_fcs = {}

        for pool in self.pools:
            if self.options['use_high_frequency_counts']:
                fcs = pool.high_frequency_log2_fold_changes
            else:
                fcs = pool.log2_fold_changes()

            all_fcs[pool.short_name] = fcs.loc[self.pool_specific_outcomes[pool.short_name], self.guides]
            
        all_fcs = pd.concat(all_fcs)

        all_fcs.index.names = ['pool_name'] + all_fcs.index.names[1:]

        return all_fcs

    def guide_log2_fold_changes(self, guide):
        ''' Look up log2 fold changes for a guide that may or may not be one of the guides
        used for clustering.
        '''
        all_fcs = {}

        for pool in self.pools:
            if self.options['use_high_frequency_counts']:
                fcs = pool.high_frequency_log2_fold_changes
            else:
                fcs = pool.log2_fold_changes()

            all_fcs[pool.short_name] = fcs.loc[self.pool_specific_outcomes[pool.short_name], guide]
            
        all_fcs = pd.concat(all_fcs)

        all_fcs.index.names = ['pool_name'] + all_fcs.index.names[1:]

        return all_fcs

def get_cluster_genes(results, guide_to_gene):
    cluster_genes = defaultdict(Counter)

    for cluster_id, guide in zip(results['cluster_assignments'], results['clustered_order']):
        gene = guide_to_gene[guide]
        cluster_genes[cluster_id][gene] += 1
    
    return cluster_genes

def assign_palette(results, axis):
    if 'cluster_assignments' in results:
        num_clusters = len(set(results['cluster_assignments']))

        if axis == 'guides':
            palette = sns.husl_palette(num_clusters)
        elif axis == 'outcomes':
            palette = sns.color_palette('muted', n_colors=num_clusters)
        else:
            raise ValueError(axis)

        results['palette'] = palette
        grey = matplotlib.colors.to_rgb('silver')
        cluster_colors = {i: palette[i] for i in range(num_clusters)}
        results['cluster_colors'] = cluster_colors

        results['colors'] = {key: cluster_colors.get(i, grey) for key, i in zip(results['clustered_order'], results['cluster_assignments'])}