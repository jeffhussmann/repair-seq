import copy
from collections import defaultdict, Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
import pandas as pd

import hits.visualize
import hits.utilities
import ddr.cluster
import ddr.visualize.outcome_diagrams

memoized_property = hits.utilities.memoized_property

correlation_cmap = copy.copy(plt.get_cmap('PuOr_r'))
correlation_cmap.set_bad('white')

fold_changes_cmap = plt.get_cmap('RdBu_r')

class Clustermap:
    def __init__(self,
                 fig=None,
                 diagram_ax_rectangle=None,
                 manual_guides=None,
                 manual_outcomes=None,
                 **options,
                ):
        '''
        diagram_ax_rectangle: rectangle (in figure fraction coords) for diagrams to occupy in existing figure
        '''
        options.setdefault('upside_down', False)
        options.setdefault('guide_library', None)
        options.setdefault('guide_method', 'hierarchical')
        options.setdefault('guide_kwargs', {})
        options.setdefault('outcome_method', 'hierarchical')
        options.setdefault('outcome_kwargs', {})
        options.setdefault('diagram_kwargs', {})
        options.setdefault('gene_text_size', 14)

        options['diagram_kwargs'].setdefault('window', (-30, 30))

        self.options = options

        self.inches_per_guide = 14 / 100

        self.manual_guides = manual_guides
        self.manual_outcomes = manual_outcomes

        self.num_guides = len(self.guides)
        self.num_outcomes = len(self.outcomes)

        self.axs = {}

        self.width_inches = {}
        self.height_inches = {}

        self.x0 = {}
        self.x1 = {}
        self.y0 = {}
        self.y1 = {}

        self.fig = fig

        if self.fig is None:
            # want 1 nt of diagram to be 3/4ths as wide as 1 guide of heatmap
            window_start, window_end = self.options['diagram_kwargs']['window']
            window_size = window_end - window_start + 1
            self.width_inches['diagrams'] = self.inches_per_guide * window_size  * 0.75
            self.height_inches['diagrams'] = self.inches_per_guide * self.num_outcomes

            self.fig = plt.figure(figsize=(self.width_inches['diagrams'], self.height_inches['diagrams']))

            diagram_ax_rectangle = (0, 0, 1, 1)

        self.fig_width_inches, self.fig_height_inches = self.fig.get_size_inches()
        self.height_per_guide = self.inches_per_guide / self.fig_height_inches
        self.width_per_guide = self.inches_per_guide / self.fig_width_inches

        self.x0['diagrams'], self.y0['diagrams'], width, height = diagram_ax_rectangle
        self.width_inches['diagrams'] = width * self.fig_width_inches
        self.height_inches['diagrams'] = height * self.fig_height_inches
            
        self.add_axes('diagrams')

        diagrams_position = self.get_position('diagrams')
        self.x0['fold changes'] = diagrams_position.x1 + diagrams_position.width * 0.02
        self.y0['fold changes'] = diagrams_position.y0
        self.width_inches['fold changes'] = self.inches_per_guide * self.num_guides
        self.height_inches['fold changes'] = self.inches_per_guide * self.num_outcomes 

        # TODO: fix weirdness in y axis here. Lining up diagrams, fold changes, and o
        # outcome similarities requires sharey and reversing outcome order in diagrams.
        self.add_axes('fold changes', sharey='diagrams')

        fold_changes_position = self.get_position('fold changes')

        gap = 0.5 * self.width_per_guide
        self.x0['outcome similarity'] = fold_changes_position.x1 + gap
        self.y0['outcome similarity'] = fold_changes_position.y0
        self.width_inches['outcome similarity'] = self.height_inches['fold changes'] / 2
        self.height_inches['outcome similarity'] = self.height_inches['fold changes']

        self.add_axes('outcome similarity')

        self.x0['guide similarity'] = fold_changes_position.x0

        gap = 0.5 * self.height_per_guide
        if self.options['upside_down']:
            self.y1['guide similarity'] = fold_changes_position.y0 - gap
        else:
            self.y0['guide similarity'] = fold_changes_position.y1 + gap

        self.width_inches['guide similarity'] = self.width_inches['fold changes']
        self.height_inches['guide similarity'] = self.width_inches['fold changes'] / 2

        self.add_axes('guide similarity')

        self.draw()

    def width(self, ax_name):
        return self.width_inches[ax_name] / self.fig_width_inches

    def height(self, ax_name):
        return self.height_inches[ax_name] / self.fig_height_inches

    def rectangle(self, ax_name):
        width = self.width(ax_name)
        height = self.height(ax_name)

        if ax_name in self.x0:
            x0 = self.x0[ax_name]
        elif ax_name in self.x1:
            x0 = self.x1[ax_name] - width
        else:
            raise ValueError(ax_name)

        if ax_name in self.y0:
            y0 = self.y0[ax_name]
        elif ax_name in self.y1:
            y0 = self.y1[ax_name] - height
        else:
            raise ValueError(ax_name)

        return (x0, y0, width, height)

    def add_axes(self, ax_name, sharex=None, sharey=None):
        ax = self.fig.add_axes(self.rectangle(ax_name), sharex=self.axs.get(sharex), sharey=self.axs.get(sharey))
        self.axs[ax_name] = ax
        return ax

    def get_position(self, ax_name):
        return self.axs[ax_name].get_position()

    def draw_diagrams(self):
        ax = self.axs['diagrams']

        # See TODO comment above on reversing here.
        ddr.visualize.outcome_diagrams.plot(self.clustered_outcomes[::-1],
                                            self.target_info,
                                            ax=ax,
                                            **self.options['diagram_kwargs'],
                                           )

        ax.set_title('distance from cut site (nts)', size=10, pad=15)
        ax.set_position(self.rectangle('diagrams'))

    def draw_fold_changes(self):
        ax = self.axs['fold changes']

        heatmap_im = ax.imshow(self.clustered_fold_changes, cmap=fold_changes_cmap, vmin=-2, vmax=2, interpolation='none')

        ax.axis('off')

        for x, guide in enumerate(self.clustered_fold_changes.columns):
            color = 'black'
            ax.annotate(guide,
                        xy=(x, 1 if self.options['upside_down'] else 0),
                        xycoords=('data', 'axes fraction'),
                        xytext=(0, 3 if self.options['upside_down'] else -3),
                        textcoords='offset points',
                        rotation=90,
                        ha='center',
                        va='bottom' if self.options['upside_down'] else 'top',
                        size=7 if color == 'black' else 8,
                        color=color,
                        weight='normal' if color == 'black' else 'bold',
                       )

    def draw_outcome_similarities(self):
        ax = self.axs['outcome similarity']

        vs = self.outcome_clustering['similarities']
        
        for side in ['left', 'top', 'bottom']:
            ax.spines[side].set_visible(False)

        ax.spines['right'].set_color('white')

        im = ax.imshow(vs, cmap=correlation_cmap, vmin=-1, vmax=1, interpolation='none')

        transform = matplotlib.transforms.Affine2D().rotate_deg(45) + ax.transData
        im.set_transform(transform)

        ax.set_xticks([])
        ax.set_yticks([])

        diagonal_length = np.sqrt(2 * len(vs)**2)

        ax.set_xlim(0, diagonal_length / 2)
        ax.set_ylim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diagonal_length)

    def draw_guide_similarities(self):
        ax = self.axs['guide similarity']

        vs = self.guide_clustering['similarities']

        guide_corr_im = ax.imshow(vs, cmap=correlation_cmap, vmin=-1, vmax=1, interpolation='none')

        ax.set_xticks([])
        ax.set_yticks([])

        plt.setp(ax.spines.values(), visible=False)

        transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + ax.transData

        guide_corr_im.set_transform(transform)

        diag_length = np.sqrt(2 * len(vs)**2)

        if self.options['upside_down']:
            ax.set_ylim(diag_length / 2, 0)
        else:
            ax.set_ylim(0, -diag_length / 2)

        ax.set_xlim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diag_length)

    def draw_guide_clusters(self):
        gap_before_clusters = self.height_per_guide * 6

        fold_changes_position = self.get_position('fold changes')
        self.x0['guide clusters'] = fold_changes_position.x0
        self.y1['guide clusters'] = fold_changes_position.y0 - gap_before_clusters
        self.width_inches['guide clusters'] = self.width_inches['fold changes']
        self.height_inches['guide clusters'] = self.inches_per_guide * 1

        ax = self.add_axes('guide clusters')

        assignments = self.guide_clustering['cluster_assignments']
        num_clusters = len(set(assignments))

        palette = sns.husl_palette(num_clusters)

        id_to_color = {i: palette[i] for i in range(num_clusters)}

        cluster_blocks = ddr.cluster.get_cluster_blocks(assignments)
        for cluster_id, blocks in cluster_blocks.items():
            if cluster_id == -1:
                continue

            for block_start, block_end in blocks:
                y = 1
                x_start = block_start + 0.1
                x_end = block_end + 0.9
                ax.plot([x_start, x_end],
                                       [y, y],
                                       color=id_to_color[cluster_id],
                                       linewidth=6,
                                       solid_capstyle='butt',
                                       clip_on=False,
                                      )

        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(0, len(self.guides))
        ax.axis('off')

    def draw_guide_dendrogram(self):
        clusters_position = self.get_position('guide clusters')

        gap_after_clusters = self.height_per_guide * 0.5

        self.x0['guide dendrogram'] = self.x0['fold changes']
        self.y1['guide dendrogram'] = clusters_position.y0 - gap_after_clusters

        self.height_inches['guide dendrogram'] = self.inches_per_guide * 10
        self.width_inches['guide dendrogram'] = self.width_inches['fold changes']

        ax = self.add_axes('guide dendrogram')

        sch.dendrogram(self.guide_clustering['linkage'],
                       ax=ax,
                       color_threshold=-1,
                       above_threshold_color='black',
                       orientation='bottom',
                      )

        ax.axis('off')

    def annotate_guide_clusters(self):
        assignments = self.guide_clustering['cluster_assignments']
        cluster_blocks = ddr.cluster.get_cluster_blocks(assignments)
        cluster_genes = ddr.cluster.get_cluster_genes(self.guide_clustering, self.options['guide_library'])
        cluster_colors = self.guide_clustering['cluster_colors']

        ax = self.axs['guide clusters']

        for cluster_id in cluster_blocks:
            if cluster_id == -1:
                continue

            blocks = cluster_blocks[cluster_id]
            for block in blocks:
                if block[1] - block[0] == 0:
                    continue

                genes = cluster_genes[cluster_id]
                gene_and_counts = sorted(genes.most_common(), key=lambda gene_and_count: (-gene_and_count[1], gene_and_count[0]))
                for i, (gene, count) in enumerate(gene_and_counts):
                    count_string = f' x{count}' if count > 1 else '' 
                    ax.annotate(f'{gene}{count_string}',
                                xy=(np.mean(blocks[0]) + 0.5, 0),
                                xycoords=('data', 'axes fraction'),
                                xytext=(0, -(self.options['gene_text_size'] + 1) * (i + 1)),
                                textcoords='offset points',
                                ha='center',
                                va='top',
                                color=cluster_colors[cluster_id],
                                size=self.options['gene_text_size'],
                               )

    @memoized_property
    def guide_clustering(self): 
        if self.options['guide_method'] == 'hierarchical':
            func = ddr.cluster.hierarchcical
        elif self.options['guide_method'] == 'HDBSCAN':
            func = ddr.cluster.HDBSCAN
        
        results = func(self.fold_changes, 'guides', **self.options['guide_kwargs'])
        ddr.cluster.assign_palette(results, 'guides')

        return results

    @memoized_property
    def outcome_clustering(self): 
        if self.options['outcome_method'] == 'hierarchical':
            func = ddr.cluster.hierarchcical
        elif self.options['outcome_method'] == 'HDBSCAN':
            func = ddr.cluster.HDBSCAN
        
        results = func(self.fold_changes, 'outcomes', **self.options['outcome_kwargs'])
        ddr.cluster.assign_palette(results, 'outcomes')

        return results

    @memoized_property
    def clustered_fold_changes(self):
        guide_order = self.guide_clustering['clustered_order']
        outcome_order = self.outcome_clustering['clustered_order']

        return self.fold_changes.loc[outcome_order, guide_order]

    @memoized_property
    def clustered_guides(self):
        return self.clustered_fold_changes.columns

    @memoized_property
    def clustered_outcomes(self):
        return self.clustered_fold_changes.index.values

    def draw(self):
        self.draw_diagrams()
        self.draw_fold_changes()
        self.draw_outcome_similarities()
        self.draw_guide_similarities()
        self.draw_guide_clusters()
        self.annotate_guide_clusters()

class SinglePoolClustermap(Clustermap):
    def __init__(self, pool, **kwargs):
        self.pool = pool
        self.target_info = pool.target_info
        super().__init__(**kwargs)

    @memoized_property
    def outcomes(self):
        if self.manual_outcomes is not None:
            return self.manual_outcomes
        else:
            return self.pool.canonical_outcomes

    @memoized_property
    def guides(self):
        if self.manual_guides is not None:
            return self.manual_guides
        else:
            return self.pool.canonical_active_guides

    @memoized_property
    def fold_changes(self):
        return self.pool.log2_fold_changes.loc[self.outcomes, self.guides]

class MultiplePoolClustermap(Clustermap):
    def __init__(self, pools, **kwargs):
        self.pools = pools
        self.target_info = self.pools[0].target_info
        super().__init__(**kwargs)

    @memoized_property
    def guides(self):
        if self.manual_guides is not None:
            return self.manual_guides
        else:
            active_guide_lists = defaultdict(list)

            for pool in self.pools:
                for i, guide in enumerate(pool.canonical_active_guides):
                    active_guide_lists[guide].append((pool.short_name, i))

            # Include guides that were active in at least two screens.
            # Only include up to 3 guides per gene, prioritized by average activity rank.

            gene_guides = defaultdict(list)

            for guide, active_pns in active_guide_lists.items():
                if len(active_pns) >= min(len(self.pools), 2):
                    average_rank = np.mean([rank for pn, rank in active_pns])
                    gene_guides[self.options['guide_library'].guide_to_gene[guide]].append((average_rank, guide))
                    
            filtered_guides = []
            for gene, guides in gene_guides.items():
                guides = [g for r, g in sorted(guides)]
                filtered_guides.extend(guides[:3])

            return filtered_guides

    @memoized_property
    def fold_changes(self):
        all_fcs = {}

        for pool in self.pools:
            fcs = pool.log2_fold_changes.loc[pool.canonical_outcomes, self.guides]
            all_fcs[pool.short_name] = fcs
            
        all_fcs = pd.concat(all_fcs)

        return all_fcs

    @memoized_property
    def outcomes(self):
        return [(c, s, d) for pn, c, s, d in self.fold_changes.index.values]

    @memoized_property
    def clustered_outcomes(self):
        return [(c, s, d) for pn, c, s, d in self.clustered_fold_changes.index.values]

def clustermap(pool,
               outcomes=40,
               guides=100,
               diagram_ax_rectangle=None,
               fig=None,
               upside_down=False,
               diagram_kwargs=None,
               fixed_guide='none',
               cluster='both',
               gene_to_color=None,
               min_UMIs=None,
               draw_colorbars=True,
               num_guide_clusters=10,
               num_outcome_clusters=10,
               draw_outcome_dendrogram=True,
               draw_guide_dendrogram=True,
               guide_cluster_label_offsets=None,
               use_hdbscan=True,
              ):

    if diagram_kwargs is None:
        diagram_kwargs = {}

    if gene_to_color is None:
        gene_to_color = {}

    if guide_cluster_label_offsets is None:
        guide_cluster_label_offsets = {}

    correlation_cmap = copy.copy(plt.get_cmap('PuOr_r'))
    correlation_cmap.set_bad('white')

    fc_cmap = plt.get_cmap('RdBu_r')

    cluster_results = ddr.cluster.hierarchcical_old(pool,
                                          outcomes,
                                          guides,
                                          method='average',
                                          fixed_guide=fixed_guide,
                                          min_UMIs=min_UMIs,
                                          num_outcome_clusters=num_outcome_clusters,
                                          num_guide_clusters=num_guide_clusters,
                                         )
    
    if use_hdbscan:
        hdbscan_results = ddr.cluster.hdbscan(pool, outcomes, guides, cluster_selection_epsilon=0.3)
        cluster_results.update(hdbscan_results)

    if cluster == 'both' or cluster == 'guides':
        final_guide_order = cluster_results['clustered_guide_order']
    else:
        final_guide_order = sorted(cluster_results['clustered_guide_order'])

    if cluster == 'both' or cluster == 'outcomes':
        final_outcome_order = cluster_results['clustered_outcome_order']
    else:
        final_outcome_order = outcomes

    num_guides = len(final_guide_order)
    num_outcomes = len(final_outcome_order)

    window_start, window_end = diagram_kwargs.get('window', (-30, 30))
    window_size = window_end - window_start + 1

    inches_per_guide = 14 / 100

    fc_ax_width_inches = inches_per_guide * num_guides
    fc_ax_height_inches = inches_per_guide * num_outcomes

    # want 1 nt of diagram to be 3/4ths as wide as 1 guide of heatmap
    diagram_width_inches = inches_per_guide * window_size  * 0.75
    diagram_height_inches = fc_ax_height_inches

    axs = {}

    if diagram_ax_rectangle is None:
        fig, diagram_ax = plt.subplots(figsize=(diagram_width_inches, diagram_height_inches))
        diagram_ax_rectangle = (0, 0, 1, 1)
    else:
        diagram_ax = fig.add_axes(diagram_ax_rectangle)
    
    axs['diagrams'] = diagram_ax

    fig_width_inches, fig_height_inches = fig.get_size_inches()

    height_per_guide = inches_per_guide / fig_height_inches
    width_per_guide = inches_per_guide / fig_width_inches

    # Draw outcome diagrams.

    ddr.visualize.outcome_diagrams.plot(final_outcome_order[::-1],
                                        pool.target_info,
                                        ax=diagram_ax,
                                        **diagram_kwargs,
                                       )

    diagram_ax.set_title('distance from cut site (nts)', size=10, pad=15)
    diagram_ax.set_position(diagram_ax_rectangle)

    diagram_ax_p = diagram_ax.get_position()

    # Draw fold-change heatmap.

    fc_x0 = diagram_ax_p.x1 + diagram_ax_p.width * 0.02
    fc_y0 = diagram_ax_p.y0
    fc_width = fc_ax_width_inches / fig_width_inches
    fc_height = fc_ax_height_inches / fig_height_inches

    fc_rectangle = (fc_x0, fc_y0, fc_width, fc_height)

    fc_ax = fig.add_axes(fc_rectangle, sharey=diagram_ax)
    axs['fold_changes'] = fc_ax

    fcs = pool.log2_fold_changes_full_arguments('perfect', fixed_guide)[fixed_guide].loc[final_outcome_order, final_guide_order]

    heatmap_kwargs = dict(cmap=fc_cmap, vmin=-2, vmax=2, interpolation='none')
    heatmap_im = fc_ax.imshow(fcs, **heatmap_kwargs)

    fc_ax.axis('off')

    fc_ax_p = fc_ax.get_position()

    # Draw outcome correlations.

    corrs = cluster_results['outcome_correlations']
    
    gap = 0.5 * width_per_guide
    outcome_corr_x0 = fc_ax_p.x1 + gap
    outcome_corr_y0 = fc_ax_p.y0
    outcome_corr_height = fc_ax_p.height
    outcome_corr_width = outcome_corr_height * fig_height_inches / 2 / fig_width_inches

    outcome_corr_rectangle = (outcome_corr_x0, outcome_corr_y0, outcome_corr_width, outcome_corr_height)

    outcome_corr_ax = fig.add_axes(outcome_corr_rectangle)
    axs['outcome_correlations'] = outcome_corr_ax

    for side in ['left', 'top', 'bottom']:
        outcome_corr_ax.spines[side].set_visible(False)

    outcome_corr_ax.spines['right'].set_color('white')

    mask = np.tri(len(corrs), k=-1)
    masked = np.ma.array(corrs)

    outcome_corr_im = outcome_corr_ax.imshow(masked, cmap=correlation_cmap, vmin=-1, vmax=1, interpolation='none')

    transform = matplotlib.transforms.Affine2D().rotate_deg(45) + outcome_corr_ax.transData
    outcome_corr_im.set_transform(transform)

    outcome_corr_ax.set_xticks([])
    outcome_corr_ax.set_yticks([])

    diagonal_length = np.sqrt(2 * len(corrs)**2)

    outcome_corr_ax.set_xlim(0, diagonal_length / 2)
    outcome_corr_ax.set_ylim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diagonal_length)

    # Draw guide correlations.

    corrs = cluster_results['guide_correlations']

    guide_corr_x0 = fc_ax_p.x0

    guide_corr_width = fc_ax_p.width
    guide_corr_height = guide_corr_width * fig_width_inches / 2 / fig_height_inches

    gap = height_per_guide * 0.5
    if upside_down:
        guide_corr_y0 = fc_ax_p.y0 - gap - guide_corr_height
    else:
        guide_corr_y0 = fc_ax_p.y1 + gap

    guide_corr_rectangle = (guide_corr_x0, guide_corr_y0, guide_corr_width, guide_corr_height)
    guide_corr_ax = fig.add_axes(guide_corr_rectangle)
    axs['guide_correlations'] = guide_corr_ax

    if upside_down:
        mask = np.triu(np.ones((len(corrs), len(corrs))), k=1)
    else:
        mask = np.tri(len(corrs), k=-1)

    masked = np.ma.array(corrs, mask=mask)

    guide_corr_im = guide_corr_ax.imshow(masked, cmap=correlation_cmap, vmin=-1, vmax=1, interpolation='none')

    guide_corr_ax.set_xticks([])
    guide_corr_ax.set_yticks([])

    plt.setp(guide_corr_ax.spines.values(), visible=False)

    transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + guide_corr_ax.transData

    guide_corr_im.set_transform(transform)

    diag_length = np.sqrt(2 * len(corrs)**2)

    if upside_down:
        guide_corr_ax.set_ylim(diag_length / 2, 0)
    else:
        guide_corr_ax.set_ylim(0, -diag_length / 2)

    guide_corr_ax.set_xlim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diag_length)

    for i, guide in enumerate(final_guide_order):
        x = i
        gene = pool.variable_guide_library.guide_to_gene[guide]
        color = gene_to_color.get(gene, 'black')
        fc_ax.annotate(guide,
                       xy=(x, 1 if upside_down else 0),
                       xycoords=('data', 'axes fraction'),
                       xytext=(0, 3 if upside_down else -3),
                       textcoords='offset points',
                       rotation=90,
                       ha='center',
                       va='bottom' if upside_down else 'top',
                       size=7 if color == 'black' else 8,
                       color=color,
                       weight='normal' if color == 'black' else 'bold',
                      )

    if draw_guide_dendrogram:
        gap_before_clusters = height_per_guide * 6
        guide_clusters_height = height_per_guide * 1
        guide_clusters_width = fc_ax_p.width

        guide_clusters_x0 = fc_ax_p.x0
        guide_clusters_y0 = fc_ax_p.y0 - gap_before_clusters - guide_clusters_height

        guide_clusters_rectangle = (guide_clusters_x0, guide_clusters_y0, guide_clusters_width, guide_clusters_height)
        guide_clusters_ax = fig.add_axes(guide_clusters_rectangle)
        axs['guide_clusters'] = guide_clusters_ax

        num_clusters = len(set(cluster_results['guide_cluster_assignments']))

        palette = sns.husl_palette(num_clusters)

        id_to_color = {i: palette[i] for i in range(num_clusters)}

        cluster_blocks = ddr.cluster.get_cluster_blocks(cluster_results['guide_cluster_assignments'])
        for cluster_id, blocks in cluster_blocks.items():
            if cluster_id == -1:
                continue

            for block_start, block_end in blocks:
                y = 1
                x_start = block_start + 0.1
                x_end = block_end + 0.9
                guide_clusters_ax.plot([x_start, x_end],
                                       [y, y],
                                       color=id_to_color[cluster_id],
                                       linewidth=6,
                                       solid_capstyle='butt',
                                       clip_on=False,
                                      )

        guide_clusters_ax.set_ylim(0.5, 1.5)
        guide_clusters_ax.set_xlim(0, num_guides)
        guide_clusters_ax.axis('off')

        gene_text_size = 14

        for cluster_id in cluster_blocks:
            if cluster_id == -1:
                continue

            blocks = cluster_blocks[cluster_id]
            for block in blocks:
                if block[1] - block[0] == 0:
                    continue

                genes = cluster_results['cluster_genes'][cluster_id]
                gene_and_counts = sorted(genes.most_common(), key=lambda gene_and_count: (-gene_and_count[1], gene_and_count[0]))
                for i, (gene, count) in enumerate(gene_and_counts):
                    count_string = f' x{count}' if count > 1 else '' 
                    guide_clusters_ax.annotate(f'{gene}{count_string}',
                                xy=(np.mean(blocks[0]) + 0.5, 0),
                                xycoords=('data', 'axes fraction'),
                                xytext=(0, -(gene_text_size + 1) * (i + 1 + guide_cluster_label_offsets.get(cluster_id, 0))),
                                textcoords='offset points',
                                ha='center',
                                va='top',
                                color=id_to_color[cluster_id],
                                size=gene_text_size,
                            )

        #guide_clusters_p = guide_clusters_ax.get_position()

        #guide_dendro_height = height_per_guide * 10
        #guide_dendro_width = fc_ax_p.width

        #gap_after_clusters = height_per_guide * 0.5

        #guide_dendro_x0 = fc_ax_p.x0
        #guide_dendro_y0 = guide_clusters_p.y0 - gap_after_clusters - guide_dendro_height

        #guide_dendro_rectangle = (guide_dendro_x0, guide_dendro_y0, guide_dendro_width, guide_dendro_height)
        #guide_dendro_ax = fig.add_axes(guide_dendro_rectangle)
        #axs['guide_dendro'] = guide_dendro_ax

        #scipy.cluster.hierarchy.dendrogram(cluster_results['guide_linkage'],
        #                                   ax=guide_dendro_ax,
        #                                   color_threshold=-1,
        #                                   above_threshold_color='black',
        #                                   orientation='bottom',
        #                                  ) 

        #guide_dendro_ax.axis('off')

    if draw_outcome_dendrogram:
        gap_before_clusters = width_per_guide * 0.5
        outcome_clusters_height = fc_ax_p.height
        outcome_clusters_width = width_per_guide * 1

        outcome_clusters_x0 = diagram_ax_p.x0 - gap_before_clusters - outcome_clusters_width
        outcome_clusters_y0 = diagram_ax_p.y0

        outcome_clusters_rectangle = (outcome_clusters_x0, outcome_clusters_y0, outcome_clusters_width, outcome_clusters_height)
        outcome_clusters_ax = fig.add_axes(outcome_clusters_rectangle)
        axs['outcome_clusters'] = outcome_clusters_ax

        num_clusters = len(set(cluster_results['outcome_cluster_assignments']))

        palette = sns.color_palette('colorblind', n_colors=num_clusters)

        id_to_color = {i: palette[i] for i in range(num_clusters)}

        cluster_blocks = ddr.cluster.get_cluster_blocks(cluster_results['outcome_cluster_assignments'])
        for cluster_id, blocks in cluster_blocks.items():
            if cluster_id == -1:
                continue

            for block_start, block_end in blocks:
                y_start = block_start + 0.1
                y_end = block_end + 0.9
                x = 1
                outcome_clusters_ax.plot([x, x],
                                         [y_start, y_end],
                                         color=id_to_color[cluster_id],
                                         linewidth=6,
                                         solid_capstyle='butt',
                                         clip_on=False,
                                        )

        outcome_clusters_ax.set_xlim(0.5, 1.5)
        outcome_clusters_ax.set_ylim(0, num_outcomes)
        outcome_clusters_ax.axis('off')

        outcome_clusters_p = outcome_clusters_ax.get_position()

        outcome_dendro_height = diagram_ax_p.height
        outcome_dendro_width = width_per_guide * 10

        gap_after_clusters = height_per_guide * 0.5

        outcome_dendro_x0 = outcome_clusters_p.x0 - gap_after_clusters - outcome_dendro_width
        outcome_dendro_y0 = diagram_ax_p.y0

        outcome_dendro_rectangle = (outcome_dendro_x0, outcome_dendro_y0, outcome_dendro_width, outcome_dendro_height)
        outcome_dendro_ax = fig.add_axes(outcome_dendro_rectangle)
        axs['outcome_dendro'] = outcome_dendro_ax

        scipy.cluster.hierarchy.dendrogram(cluster_results['outcome_linkage'],
                                           ax=outcome_dendro_ax,
                                           color_threshold=-1,
                                           above_threshold_color='black',
                                           orientation='left',
                                          ) 

        outcome_dendro_ax.axis('off')

    else:
        # Draw non-targeting fractions.

        frequency_width = width_per_guide * 20
        frequency_gap = width_per_guide * 1

        frequency_x0 = diagram_ax_p.x0 - frequency_gap - frequency_width
        frequency_y0 = diagram_ax_p.y0
        frequency_height = diagram_ax_p.height
        
        frequency_rectangle = (frequency_x0, frequency_y0, frequency_width, frequency_height)

        frequency_ax = fig.add_axes(frequency_rectangle, sharey=fc_ax)
        axs['frequencies'] = frequency_ax

        frequencies = pool.non_targeting_fractions('perfect', 'none').loc[final_outcome_order]

        xs = np.log10(frequencies)
        ys = np.arange(len(frequencies))
        frequency_ax.plot(xs, ys, '.', markeredgewidth=0, markersize=10, alpha=0.9, clip_on=False, color='black')

        x_lims = np.log10(np.array([2e-3, 2e-1]))

        for exponent in [3, 2, 1]:
            xs = np.log10(np.arange(1, 10) * 10**-exponent)        
            for x in xs:
                if x_lims[0] <= x <= x_lims[1]:
                    frequency_ax.axvline(x, color='black', alpha=0.07, clip_on=False)

        x_ticks = [x for x in [2e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1] if x_lims[0] <= np.log10(x) <= x_lims[1]]

        frequency_ax.set_xticks(np.log10(x_ticks))
        frequency_ax.set_xticklabels([f'{100 * x:g}' for x in x_ticks])

        for side in ['left', 'right', 'bottom']:
            frequency_ax.spines[side].set_visible(False)

        frequency_ax.xaxis.tick_top()
        frequency_ax.set_xlim(*x_lims)
        frequency_ax.invert_xaxis()

        frequency_ax.set_title('percentage of outcomes\nin cells containing\nnon-targeting guides', size=10, pad=15)

    if draw_colorbars:
        corr_ax_p = outcome_corr_ax.get_position()
        # Draw correlation colorbar.
        width = corr_ax_p.width * 4 / num_outcomes
        height = corr_ax_p.height * 0.4
        cbar_rectangle = (corr_ax_p.x0 + corr_ax_p.width * 0.5 - width * 0.5,
                          corr_ax_p.y1,
                          width,
                          height,
                         )
        cbar_ax = fig.add_axes(cbar_rectangle)
        cbar = fig.colorbar(guide_corr_im, cax=cbar_ax, ticks=[-1, 0, 1])
        cbar_ax.annotate('correlation\nbetween\noutcome\nredistribution\nprofiles',
                         xy=(1, 0.5),
                         xycoords='axes fraction',
                         xytext=(60, 0),
                         textcoords='offset points',
                         ha='center',
                         va='center',
                        )

        cbar.outline.set_alpha(0.1)

        #ddr.visualize.heatmap.add_fold_change_colorbar(fig, heatmap_im, -0.05, 0.4, 0.15, 0.02)

        if upside_down:
            title_kwargs = dict(
                xy=(0.5, 0),
                xytext=(0, -15),
                va='top',
            )
        else:
            title_kwargs = dict(
                xy=(0.5, 1),
                xytext=(0, 15),
                va='bottom',
            )

        guide_corr_ax.annotate(pool.group,
                xycoords='axes fraction',
                textcoords='offset points',
                ha='center',
                **title_kwargs,
        )

    return fig, axs, cluster_results
