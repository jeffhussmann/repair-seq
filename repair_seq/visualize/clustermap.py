from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pandas as pd

import hits.visualize
import hits.utilities
import knock_knock.outcome
import repair_seq.cluster
import repair_seq.visualize
import repair_seq.visualize.outcome_diagrams
import repair_seq.visualize.heatmap

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
        options.setdefault('draw_outcome_clusters', False)
        options.setdefault('draw_guide_clusters', True)
        options.setdefault('draw_outcome_similarities', True)
        options.setdefault('draw_guide_similarities', True)
        options.setdefault('draw_colorbars', True)
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

        self.ims = {}

        self.fig = fig

        if self.fig is None:
            # want 1 nt of diagram to be 3/4ths as wide as 1 guide of heatmap
            window_start, window_end = self.options['diagram_kwargs']['window']
            window_size = window_end - window_start + 1
            self.width_inches['diagrams'] = self.inches_per_guide * window_size  * 0.75
            self.height_inches['diagrams'] = self.inches_per_guide * self.num_outcomes

            self.fig = plt.figure(figsize=(self.width_inches['diagrams'], self.height_inches['diagrams']))

            diagram_ax_rectangle = (0, 0, 1, 1)

        self.diagram_ax_rectangle = diagram_ax_rectangle

        self.fig_width_inches, self.fig_height_inches = self.fig.get_size_inches()
        self.height_per_guide = self.inches_per_guide / self.fig_height_inches
        self.width_per_guide = self.inches_per_guide / self.fig_width_inches

        self.draw()

    def draw(self):
        self.draw_diagrams()

        self.draw_fold_changes()

        if self.options['draw_outcome_similarities']:
            self.draw_outcome_similarities()

        if self.options['draw_guide_similarities']:
            self.draw_guide_similarities()

        if self.options['draw_guide_clusters']:
            self.draw_guide_clusters()
            self.annotate_guide_clusters()

        if self.options['draw_outcome_clusters']:
            self.draw_outcome_clusters()

        if self.options['draw_colorbars']:
            self.draw_colorbars()

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
        self.x0['diagrams'], self.y0['diagrams'], width, height = self.diagram_ax_rectangle
        self.width_inches['diagrams'] = width * self.fig_width_inches
        self.height_inches['diagrams'] = height * self.fig_height_inches
            
        ax = self.add_axes('diagrams')

        # See TODO comment in fold changes on reversing here.

        outcomes = self.clusterer.clustered_outcomes[::-1]

        repair_seq.visualize.outcome_diagrams.plot(outcomes,
                                            self.clusterer.pn_to_target_info,
                                            ax=ax,
                                            replacement_text_for_complex={
                                                'genomic insertion, hg19, <=75 nts': 'capture of genomic sequence ≤75 nts',
                                                'genomic insertion, hg19, >75 nts': 'capture of genomic sequence >75 nts',
                                            },
                                            **self.options['diagram_kwargs'],
                                           )

        ax.set_position(self.rectangle('diagrams'))

    def draw_fold_changes(self):
        diagrams_position = self.get_position('diagrams')
        self.x0['fold changes'] = diagrams_position.x1 + diagrams_position.width * 0.02
        self.y0['fold changes'] = diagrams_position.y0
        self.width_inches['fold changes'] = self.inches_per_guide * self.num_guides
        self.height_inches['fold changes'] = self.inches_per_guide * self.num_outcomes 

        # TODO: fix weirdness in y axis here. Lining up diagrams, fold changes, and
        # outcome similarities requires sharey and reversing outcome order in diagrams.
        ax = self.add_axes('fold changes', sharey='diagrams')

        im = ax.imshow(self.clusterer.clustered_log2_fold_changes,
                       cmap=repair_seq.visualize.fold_changes_cmap,
                       vmin=-2, vmax=2,
                       interpolation='none',
                      )

        self.ims['fold changes'] = im

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
        fold_changes_position = self.get_position('fold changes')

        gap = 0.5 * self.width_per_guide
        self.x0['outcome similarity'] = fold_changes_position.x1 + gap
        self.y0['outcome similarity'] = fold_changes_position.y0
        self.width_inches['outcome similarity'] = self.height_inches['fold changes'] / 2
        self.height_inches['outcome similarity'] = self.height_inches['fold changes']

        ax = self.add_axes('outcome similarity')

        vs = self.clusterer.outcome_clustering['similarities']
        
        for side in ['left', 'top', 'bottom']:
            ax.spines[side].set_visible(False)

        ax.spines['right'].set_color('white')

        im = ax.imshow(vs,
                       cmap=repair_seq.visualize.correlation_cmap,
                       vmin=-1,
                       vmax=1,
                       interpolation='none',
                      )

        self.ims['outcome similarities'] = im

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

        im = ax.imshow(vs,
                       cmap=repair_seq.visualize.correlation_cmap,
                       vmin=-1,
                       vmax=1,
                       interpolation='none',
                      )

        self.ims['guide similarities'] = im

        ax.set_xticks([])
        ax.set_yticks([])

        plt.setp(ax.spines.values(), visible=False)

        transform = matplotlib.transforms.Affine2D().rotate_deg(-45) + ax.transData

        im.set_transform(transform)

        diag_length = np.sqrt(2 * len(vs)**2)

        if self.options['upside_down']:
            ax.set_ylim(diag_length / 2, 0)
        else:
            ax.set_ylim(0, -diag_length / 2)

        ax.set_xlim(-np.sqrt(2) / 2, -np.sqrt(2) / 2 + diag_length)

    def draw_guide_clusters(self):
        guide_clustering = self.clusterer.guide_clustering

        gap_before_clusters = self.height_per_guide * 10

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

        cluster_blocks = repair_seq.cluster.get_cluster_blocks(assignments)
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

    def draw_guide_cell_cycle_effects(self):
        ax_key = 'guide cell cycle'
        guide_order = self.clusterer.guide_clustering['clustered_order']
        cell_cycle_fold_changes = self.clusterer.guide_library.cell_cycle_log2_fold_changes[guide_order]

        gap_between = self.height_per_guide * 6

        fold_changes_position = self.get_position('fold changes')

        self.x0[ax_key] = fold_changes_position.x0

        if self.options['upside_down']:
            self.y0[ax_key] = fold_changes_position.y1 + gap_between
        else:
            self.y1[ax_key] = fold_changes_position.y0 - gap_between

        self.width_inches[ax_key] = self.width_inches['fold changes']
        self.height_inches[ax_key] = self.inches_per_guide * 3

        ax = self.add_axes(ax_key)

        im = ax.imshow(cell_cycle_fold_changes,
                       cmap=repair_seq.visualize.cell_cycle_cmap,
                       vmin=-1, vmax=1,
                       interpolation='none',
                      )

        for y, phase in enumerate(cell_cycle_fold_changes.index):
            ax.annotate(phase,
                        xy=(0, y),
                        xycoords=('axes fraction', 'data'),
                        xytext=(-5, 0),
                        textcoords='offset points',
                        ha='right',
                        va='center',
                       )

        ax.axis('off')

        colorbar_key = f'{ax_key} colorbar'
        heatmap_position = self.get_position(ax_key)

        self.x0[colorbar_key] = heatmap_position.x1 + self.width_per_guide * 5
        self.y0[colorbar_key] = heatmap_position.y0 + self.height_per_guide * 0.75

        self.width_inches[colorbar_key] = self.inches_per_guide * 5
        self.height_inches[colorbar_key] = self.inches_per_guide * 1.5

        colorbar_ax = self.add_axes(colorbar_key)
        colorbar = plt.colorbar(mappable=im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label(f'log$_2$ fold change\nin cell-cycle phase\noccupancy')

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

        cluster_blocks = repair_seq.cluster.get_cluster_blocks(outcome_clustering['cluster_assignments'])
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

    def draw_outcome_categories(self):
        diagrams_position = self.get_position('diagrams')

        gap_before_categories = self.width_per_guide * 0.5

        self.x1['outcome categories'] = diagrams_position.x0 - gap_before_categories
        self.y0['outcome categories'] = diagrams_position.y0

        self.height_inches['outcome categories'] = self.height_inches['fold changes']
        self.width_inches['outcome categories'] = self.inches_per_guide * 2

        ax = self.add_axes('outcome categories')

        full_categories = defaultdict(list)

        outcomes = self.clusterer.clustered_outcomes[::-1]

        for pn, c, s, d in outcomes:
            ti = self.clusterer.pn_to_target_info[pn]
            if c == 'deletion':
                deletion = knock_knock.outcome.DeletionOutcome.from_string(d).undo_anchor_shift(ti.anchor)
                directionality = deletion.classify_directionality(ti)
                full_category = f'{c}, {directionality}'
            else:
                full_category = c
                
            full_categories[full_category].append((pn, c, s, d))

        x = 0

        effector = self.clusterer.target_info.effector.name
        category_display_order = repair_seq.visualize.category_display_order[effector]
        category_colors = repair_seq.visualize.category_colors[effector]

        for cat in category_display_order:
            cat_outcomes = full_categories[cat]
            if len(cat_outcomes) > 0:
                indices = sorted([len(outcomes) - 1 - outcomes.index(outcome) for outcome in cat_outcomes])
                connected_blocks = []
                current_block_start = indices[0]
                current_idx = indices[0]

                for next_idx in indices[1:]:
                    if next_idx - current_idx > 1:
                        block = (current_block_start, current_idx)
                        connected_blocks.append(block)
                        current_block_start = next_idx
                        
                    current_idx = next_idx

                # Close off last block
                block = (current_block_start, current_idx)
                connected_blocks.append(block) 

                for first, last in connected_blocks:
                    ax.plot([x, x], [first - 0.4, last + 0.4],
                            linewidth=6,
                            color=category_colors[cat],
                            clip_on=False,
                            solid_capstyle='butt',
                           )

                x -= 1

        ax.set_xlim(x - 1, 1)
        ax.set_ylim(-0.5, len(outcomes) - 0.5)
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
        cluster_blocks = repair_seq.cluster.get_cluster_blocks(assignments)
        cluster_genes = repair_seq.cluster.get_cluster_genes(guide_clustering, self.clusterer.guide_to_gene)
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

    def draw_colorbars(self):

        # Similary colorbar.

        self.x0['similarity colorbar'] = self.x0['guide similarity']
        self.y0['similarity colorbar'] = self.y0['guide similarity'] + 20 * self.height_per_guide

        self.width_inches['similarity colorbar'] = self.inches_per_guide * 1.5
        self.height_inches['similarity colorbar'] = self.inches_per_guide * 18

        ax = self.add_axes('similarity colorbar')

        cbar = plt.colorbar(self.ims['guide similarities'], cax=ax, orientation='vertical', ticks=[-1, 0, 1])

        cbar.outline.set_alpha(0.1)

        ax.annotate('correlation\nbetween\nrepair outcome\nredistribution\nprofiles',
                    xy=(1, 0.5),
                    xycoords='axes fraction',
                    xytext=(18, 0),
                    textcoords='offset points',
                    ha='left',
                    va='center',
                    size=12,
                   )

        # Fold changes colorbar.

        diagrams_position = self.get_position('diagrams')
        self.x0['fold changes colorbar'] = diagrams_position.x0 + 10 * self.width_per_guide
        self.y0['fold changes colorbar'] = diagrams_position.y1 + 10 * self.height_per_guide

        self.width_inches['fold changes colorbar'] = self.inches_per_guide * 15
        self.height_inches['fold changes colorbar'] = self.inches_per_guide * 1.5

        ax = self.add_axes('fold changes colorbar')

        repair_seq.visualize.heatmap.add_fold_change_colorbar(self.fig, self.ims['fold changes'], cbar_ax=ax, text_size=12)


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