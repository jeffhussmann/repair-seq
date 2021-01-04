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

class Clustermap:
    def __init__(self,
                 clusterer,
                 fig=None,
                 diagram_ax_rectangle=None,
                 **options,
                ):
        '''
        diagram_ax_rectangle: rectangle (in figure fraction coords) for diagrams to occupy in existing figure
        '''
        options.setdefault('upside_down', False)
        options.setdefault('guide_library', None)
        options.setdefault('diagram_kwargs', {})
        options.setdefault('gene_text_size', 14)

        options['diagram_kwargs'].setdefault('window', (-30, 30))

        self.options = options

        self.clusterer = clusterer

        self.inches_per_guide = 14 / 100

        self.num_guides = len(self.clusterer.guides)
        self.num_outcomes = len(self.clusterer.outcomes)

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
        ddr.visualize.outcome_diagrams.plot(self.clusterer.clustered_outcomes[::-1],
                                            self.clusterer.pool.target_info,
                                            ax=ax,
                                            **self.options['diagram_kwargs'],
                                           )

        ax.set_title('distance from cut site (nts)', size=10, pad=15)
        ax.set_position(self.rectangle('diagrams'))

    def draw_fold_changes(self):
        ax = self.axs['fold changes']

        heatmap_im = ax.imshow(self.clusterer.clustered_log2_fold_changes, cmap=fold_changes_cmap, vmin=-2, vmax=2, interpolation='none')

        ax.axis('off')

        for x, guide in enumerate(self.clusterer.clustered_guides):
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

        vs = self.clusterer.outcome_clustering['similarities']
        
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
        fold_changes_position = self.get_position('fold changes')

        self.x0['guide similarity'] = fold_changes_position.x0

        gap = 0.5 * self.height_per_guide
        if self.options['upside_down']:
            self.y1['guide similarity'] = fold_changes_position.y0 - gap
        else:
            self.y0['guide similarity'] = fold_changes_position.y1 + gap

        self.width_inches['guide similarity'] = self.width_inches['fold changes']
        self.height_inches['guide similarity'] = self.width_inches['fold changes'] / 2

        ax = self.add_axes('guide similarity')

        vs = self.clusterer.guide_clustering['similarities']

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
        guide_clustering = self.clusterer.guide_clustering

        gap_before_clusters = self.height_per_guide * 6

        fold_changes_position = self.get_position('fold changes')

        self.x0['guide clusters'] = fold_changes_position.x0

        if self.options['upside_down']:
            self.y0['guide clusters'] = fold_changes_position.y1 + gap_before_clusters
        else:
            self.y1['guide clusters'] = fold_changes_position.y0 - gap_before_clusters

        self.width_inches['guide clusters'] = self.width_inches['fold changes']
        self.height_inches['guide clusters'] = self.inches_per_guide * 1

        ax = self.add_axes('guide clusters')

        assignments = guide_clustering['cluster_assignments']

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
                        color=guide_clustering['cluster_colors'][cluster_id],
                        linewidth=6,
                        solid_capstyle='butt',
                        clip_on=False,
                       )

        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(0, len(self.clusterer.guides))
        ax.axis('off')

    def draw_guide_dendrogram(self):
        clusters_position = self.get_position('guide clusters')

        gap_after_clusters = self.height_per_guide * 0.5

        self.x0['guide dendrogram'] = self.x0['fold changes']
        self.y1['guide dendrogram'] = clusters_position.y0 - gap_after_clusters

        self.height_inches['guide dendrogram'] = self.inches_per_guide * 10
        self.width_inches['guide dendrogram'] = self.width_inches['fold changes']

        ax = self.add_axes('guide dendrogram')

        sch.dendrogram(self.clusterer.guide_clustering['linkage'],
                       ax=ax,
                       color_threshold=-1,
                       above_threshold_color='black',
                       orientation='bottom',
                      )

        ax.axis('off')

    def draw_outcome_clusters(self):
        outcome_clustering = self.clusterer.outcome_clustering

        diagrams_position = self.get_position('diagrams')

        gap_before_clusters = self.width_per_guide * 0.5

        self.x1['outcome clusters'] = diagrams_position.x0 - gap_before_clusters
        self.y0['outcome clusters'] = diagrams_position.y0

        self.height_inches['outcome clusters'] = self.height_inches['fold changes']
        self.width_inches['outcome clusters'] = self.inches_per_guide * 0.5

        ax = self.add_axes('outcome clusters')

        cluster_blocks = ddr.cluster.get_cluster_blocks(outcome_clustering['cluster_assignments'])
        for cluster_id, blocks in cluster_blocks.items():
            if cluster_id == -1:
                continue

            for block_start, block_end in blocks:
                y_start = block_start + 0.1
                y_end = block_end + 0.9
                x = 1
                ax.plot([x, x],
                        [y_start, y_end],
                        color=outcome_clustering['cluster_colors'][cluster_id],
                        linewidth=6,
                        solid_capstyle='butt',
                        clip_on=False,
                        )

        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0, len(self.clusterer.outcomes))
        ax.axis('off')

    def draw_outcome_dendrogram(self):
        clusters_position = self.get_position('outcome clusters')

        gap_after_clusters = self.width_per_guide * 0.5

        self.height_inches['outcome dendro'] = self.height_inches['outcome clusters']
        self.width_inches['outcome dendro'] = self.inches_per_guide * 10

        self.x1['outcome dendro'] = clusters_position.x0 - gap_after_clusters
        self.y0['outcome dendro'] = clusters_position.y0

        ax = self.add_axes('outcome dendro')

        sch.dendrogram(self.clusterer.outcome_clustering['linkage'],
                       ax=ax,
                       color_threshold=-1,
                       above_threshold_color='black',
                       orientation='left',
                      )

        ax.axis('off')

    def annotate_guide_clusters(self):
        guide_clustering = self.clusterer.guide_clustering

        assignments = guide_clustering['cluster_assignments']
        cluster_blocks = ddr.cluster.get_cluster_blocks(assignments)
        cluster_genes = ddr.cluster.get_cluster_genes(guide_clustering, self.clusterer.pool.variable_guide_library)
        cluster_colors = guide_clustering['cluster_colors']

        ax = self.axs['guide clusters']

        if self.options['upside_down']:
            initial_offset = 7
            step_sign = 1
            va = 'bottom'
        else:
            initial_offset = -5
            step_sign = -1
            va = 'top'

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
                                xytext=(0, initial_offset + step_sign * (self.options['gene_text_size'] + 1) * i),
                                textcoords='offset points',
                                ha='center',
                                va=va,
                                color=cluster_colors[cluster_id],
                                size=self.options['gene_text_size'],
                               )

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