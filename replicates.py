from collections import defaultdict

import bokeh.palettes
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from scipy.stats.stats import pearsonr

import hits.utilities
memoized_property = hits.utilities.memoized_property

import knock_knock.outcome

import ddr.visualize
import ddr.visualize.clustermap
import ddr.visualize.outcome_diagrams
import ddr.pooled_screen

ALL_NON_TARGETING = ddr.pooled_screen.ALL_NON_TARGETING

class ReplicatePair:
    def __init__(self, pools, pn_pair):
        self.pools = pools
        self.pn_pair = pn_pair
        self.pn0 = pn_pair[0]
        self.pn1 = pn_pair[1]

        self.pool0 = self.pools[pn_pair[0]]
        self.pool1 = self.pools[pn_pair[1]]

        self.target_info = self.pool0.target_info

    @memoized_property
    def average_nt_fractions(self):
        nt_fracs = {pn: self.pools[pn].non_targeting_fractions for pn in self.pn_pair}
        return pd.concat(nt_fracs, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)

    @memoized_property
    def common_guides(self):
        return sorted(set(self.pool0.variable_guide_library.guides) & set(self.pool1.variable_guide_library.guides))

    @memoized_property
    def common_guides_df(self):
        return self.pool0.variable_guide_library.guides_df.loc[self.common_guides]

    @memoized_property
    def common_non_targeting_guides(self):
        return self.common_guides_df.query('gene == "negative_control"').index

    def outcomes_above_simple_threshold(self, threshold):
        return [(c, s, d) for (c, s, d), f in self.average_nt_fractions.items() if f >= threshold
                and c not in ['uncategorized', 'genomic insertion']
               ]

    def union_of_top_n_guides(self, outcomes, n):
        all_guides = set()

        for pn in self.pn_pair:
            pool = self.pools[pn]
            top_guides = pool.top_n_active_guides(outcomes, n)
            all_guides.update(top_guides)

        # Only consider guides that were present in both screens.
        all_guides = all_guides & set(self.common_guides)
        
        all_guides = sorted(all_guides)

        return all_guides

    def log2_fold_changes(self, outcomes, guides):
        return pd.concat({pn: self.pools[pn].log2_fold_changes.loc[outcomes, guides] for pn in self.pn_pair}, axis=1)

    def outcome_r_matrix(self, outcomes, guides):
        log2_fold_changes = self.log2_fold_changes(outcomes, guides)
        r_matrix = np.zeros((len(outcomes), len(outcomes)))

        for i in range(len(outcomes)):
            row = log2_fold_changes.iloc[i]
            r, p = scipy.stats.pearsonr(row[self.pn0], row[self.pn1])
            r_matrix[i, i] = r

        highest_rs = []

        for i in range(len(outcomes)):
            row_i = log2_fold_changes.iloc[i]
            for j in range(i + 1, len(outcomes)):
                row_j = log2_fold_changes.iloc[j]
                r, p = scipy.stats.pearsonr(row_i[self.pn0], row_j[self.pn0])
                r_matrix[i, j] = r
                highest_rs.append((r, i, j, log2_fold_changes.index.values[i], log2_fold_changes.index.values[j]))
                
                r, p = scipy.stats.pearsonr(row_i[self.pn1], row_j[self.pn1])
                r_matrix[j, i] = r
                
        highest_rs = sorted(highest_rs, reverse=True)

        return r_matrix, highest_rs

    def outcome_r_series(self, guides=None):
        if guides is None:
            pool = self.pools[self.pn0]
            guides = pool.variable_guide_library.guides

        outcomes = self.average_nt_fractions.iloc[:100].index.values

        log2_fold_changes = self.log2_fold_changes(outcomes, guides)

        rs = []
        ps = []

        for outcome in outcomes:
            row = log2_fold_changes.loc[outcome]
            r, p = scipy.stats.pearsonr(row[self.pn0], row[self.pn1])
            rs.append(r)
            ps.append(p)

        return pd.Series(rs, index=pd.MultiIndex.from_tuples(outcomes))

    def guide_r_matrix(self, outcomes, guides):
        log2_fold_changes = self.log2_fold_changes(outcomes, guides)
        r_matrix = np.zeros((len(guides), len(guides)))

        for i, guide in enumerate(guides):
            cols = log2_fold_changes.xs(guide, axis=1, level=1)
            r, p = scipy.stats.pearsonr(cols[self.pn0], cols[self.pn1])
            r_matrix[i, i] = r

        highest_rs = []

        for i, guide_i in enumerate(guides):
            cols_i = log2_fold_changes.xs(guide_i, axis=1, level=1)
            for j in range(i + 1, len(guides)):
                guide_j = guides[j]
                cols_j = log2_fold_changes.xs(guide_j, axis=1, level=1)

                r, p = scipy.stats.pearsonr(cols_i[self.pn0], cols_j[self.pn0])
                r_matrix[i, j] = r
                highest_rs.append((r, guides[i], guides[j]))
                
                r, p = scipy.stats.pearsonr(cols_i[self.pn1], cols_j[self.pn1])
                r_matrix[j, i] = r
                
        highest_rs = sorted(highest_rs, reverse=True)

        return pd.DataFrame(r_matrix, index=guides, columns=guides), highest_rs

    def guide_r_series(self, outcomes):
        guides = self.common_guides
        log2_fold_changes = self.log2_fold_changes(outcomes, guides)

        rs = []
        ps = []
        for guide in guides:
            cols = log2_fold_changes.xs(guide, axis=1, level=1)
            r, p = scipy.stats.pearsonr(cols[self.pn0], cols[self.pn1])
            rs.append(r)
            ps.append(p)

        return pd.Series(rs, index=guides, name='r').sort_values(ascending=False)

    def plot_outcome_diagrams_with_correlations(self, threshold=5e-3, n_guides=100, window=(-20, 20)):
        outcomes = self.outcomes_above_simple_threshold(threshold)
        guides = self.union_of_top_n_guides(outcomes, n_guides)

        r_series = self.outcome_r_series(guides=guides)
        r_matrix, _ = self.outcome_r_matrix(outcomes, guides)

        inches_per_nt = 0.1
        text_size = 7
        inches_per_outcome = 0.2
        scale = 0.5
        g = ddr.visualize.outcome_diagrams.DiagramGrid(outcomes,
                                                       self.target_info,
                                                       window=window,
                                                       cut_color='PAM',
                                                       draw_wild_type_on_top=True,
                                                       flip_if_reverse=False,
                                                       inches_per_nt=inches_per_nt * scale,
                                                       text_size=text_size * scale,
                                                       inches_per_outcome=inches_per_outcome * scale,
                                                       line_widths=0.75,
                                                       title=self.target_info.sgRNA,
                                                       title_size=10,
                                                      )

        g.add_ax('frequency',
                 width_multiple=10,
                 title='percentage of\noutcomes for\nnon-targeting\nguides',
                 title_size=7,
                )
        g.plot_on_ax('frequency', self.average_nt_fractions,
                     transform=lambda f: f * 100,
                     marker='o',
                     color='black',
                     clip_on=False,
                     markersize=1,
                     line_alpha=0.5,
                    )

        plt.setp(g.axs_by_name['frequency'].get_xticklabels(), size=6)
        g.axs_by_name['frequency'].tick_params(axis='x', which='both', pad=2)

        #g.axs_by_name['frequency'].set_xlim(np.log10(0.49e-2), np.log10(40e-2))
        #g.style_log10_frequency_ax('frequency')

        g.add_ax('correlation', gap_multiple=2, width_multiple=10,
                 title='correlation between\nreplicates in outcome\nredistribution\n across guides',
                 title_size=7,
                )

        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=ddr.visualize.correlation_cmap)

        g.plot_on_ax('correlation',
                     r_series,
                     marker=None,
                     color=sm.to_rgba(0.5),
                     clip_on=False,
                     markersize=3,
                     line_alpha=0.5,
                    )

        rs = r_series.loc[outcomes][::-1]
        colors = [tuple(row) for row in sm.to_rgba(rs)]
        g.axs_by_name['correlation'].scatter(rs, np.arange(len(rs)),
                                             color=colors, linewidths=(0,), s=8, clip_on=False,
                                             zorder=10,
                                            )

        g.axs_by_name['correlation'].set_xlim(-1, 1)
        g.axs_by_name['correlation'].set_xticks([-1, -0.5, 0, 0.5, 1])
        g.axs_by_name['correlation'].set_xticks([-0.75, -0.25, 0.25, 0.75], minor=True)
        g.axs_by_name['correlation'].tick_params(axis='x', which='both', pad=2)
        g.style_fold_change_ax('correlation')

        plt.setp(g.axs_by_name['correlation'].get_xticklabels(), size=6)

        # Note reversing of row order.
        g.add_heatmap(r_matrix[::-1], 'r',
                      cmap=ddr.visualize.correlation_cmap,
                      draw_tick_labels=False,
                      vmin=-1, vmax=1,
                      gap_multiple=2,
                     )

        full_categories, colors, category_aliases, cat_display_order = self.full_categories(outcomes)

        for cat in cat_display_order:
            cat_outcomes = full_categories[cat]
            if len(cat_outcomes) >= 0:
                g.mark_subset(cat_outcomes,
                              color=colors[cat],
                              title=category_aliases.get(cat, cat),
                              width_multiple=1.2,
                              gap_multiple=0.1,
                              size=7,
                              linewidth=3,
                             )

        return g.fig

    def plot_guide_r_matrix(self, threshold=5e-3, n_guides=100, only_genes_with_multiple=True):
        outcomes = self.outcomes_above_simple_threshold(threshold)
        guides = self.union_of_top_n_guides(outcomes, n_guides)
        
        if only_genes_with_multiple:
            gene_counts = self.common_guides_df.loc[guides, 'gene'].value_counts()
            genes_with_multiple = set(gene_counts[gene_counts > 1].index)
            guides = [guide for guide in guides if self.common_guides_df.loc[guide, 'gene'] in genes_with_multiple]

        guide_r_matrix, _ = self.guide_r_matrix(outcomes, guides)
        
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(guide_r_matrix, cmap=ddr.visualize.clustermap.correlation_cmap, vmin=-1, vmax=1)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.setp(ax.spines.values(), visible=False)
        title = '''\
        correlations between guides
        diagonal: reproducibility between replicates (correlation of guide i in rep 1 with guide i in rep 2)
        above diagonal: correlations between distinct guides in rep 1
        below diagonal: correlations between distinct guides in rep 2
        only genes with more than one active guide are included so that blocks on diagonal show within-gene consistency
        '''
        ax.set_title(title)

        return fig, guides

    def guide_guide_correlation_reproducibility(self, threshold=5e-3, n_guides=100):
        outcomes = self.outcomes_above_simple_threshold(threshold)
        guides = self.union_of_top_n_guides(outcomes, n_guides)
        
        guide_r_matrix, _ = self.guide_r_matrix(outcomes, guides)
        upper_t = np.triu_indices_from(guide_r_matrix, 1)
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(guide_r_matrix.values[upper_t], guide_r_matrix.T.values[upper_t], s=20, linewidths=(0,), alpha=0.5)

        kwargs = dict(color='black', alpha=0.3)
        ax.axhline(0, **kwargs)
        ax.axvline(0, **kwargs)
        ax.plot([-1, 1], [-1, 1], **kwargs)

        ax.set_ylabel('guide-guide correlation in replicate 2')
        ax.set_xlabel('guide-guide correlation in replicate 1')

        return fig

    def outcome_outcome_correlation_scatter(self, threshold=5e-3, n_guides=100):
        threshold = 5e-3

        outcomes = self.outcomes_above_simple_threshold(threshold)
        guides = self.union_of_top_n_guides(outcomes, n_guides)

        log2_fold_changes = self.log2_fold_changes(outcomes, guides)

        rs = {}
        for pn in self.pn_pair:
            all_corrs = log2_fold_changes[pn].T.corr()
            all_corrs.index = np.arange(len(all_corrs))
            all_corrs.columns = np.arange(len(all_corrs))
            rs[pn] = all_corrs.stack()

        rs = pd.DataFrame(rs)

        rs.index.names = ['outcome_1', 'outcome_2']

        for i in [1, 2]:
            k = f'outcome_{i}'
            rs[k] = rs.index.get_level_values(k)

        full_categories, colors, category_aliases, cat_display_order = self.full_categories(outcomes)

        cat_to_highlight = 'deletion, bidirectional'
        relevant_indices = [outcomes.index(outcome) for outcome in full_categories[cat_to_highlight]]

        df = rs.query('outcome_1 < outcome_2').copy()

        df['color'] = 'grey'
        df.loc[df['outcome_1'].isin(relevant_indices) & df['outcome_2'].isin(relevant_indices), 'color'] = colors[cat_to_highlight]

        x = self.pn0
        y = self.pn1

        fig, ax = plt.subplots(figsize=(1.5, 1.5))

        kwargs = dict(x=x, y=y, color='color', linewidths=(0,))
        ax.scatter(data=df.query('color == "grey"'), label='', alpha=0.4, s=10, **kwargs)
        ax.scatter(data=df.query('color != "grey"'), label='bidirectional deletions', s=15, **kwargs)
        
        ax.annotate('pairs of\nbidirectional\ndeletions',
                    xy=(0, 1),
                    xycoords='axes fraction',
                    xytext=(3, -3),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    color=colors[cat_to_highlight],
                    size=6,
                   )

        ax.annotate('all other pairs',
                    xy=(0, 1),
                    xycoords='axes fraction',
                    xytext=(3, -24),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    color='grey',
                    size=6,
                   )

        kwargs = dict(color='black', alpha=0.3, linewidth=0.75)
        ax.axhline(0, **kwargs)
        ax.axvline(0, **kwargs)
        ax.plot([-1, 1], [-1, 1], **kwargs)

        ticks = [-1, -0.5, 0, 0.5, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), size=6)

        ax.set_ylabel('correlation between distinct\noutcomes in replicate 2', size=6)
        ax.set_xlabel('correlation between distinct\noutcomes in replicate 1', size=6)

        return fig

    def plot_outcome_r_matrix(self, threshold=5e-3, n_guides=100):
        outcomes = self.outcomes_above_simple_threshold(threshold)
        guides = self.union_of_top_n_guides(outcomes, n_guides)
        
        outcome_r_matrix, _ = self.outcome_r_matrix(outcomes, guides)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(outcome_r_matrix, cmap=ddr.visualize.correlation_cmap, vmin=-1, vmax=1)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.setp(ax.spines.values(), visible=False)

        title = '''\
        correlations between outcomes
        diagonal: reproducibility between replicates (correlation of outcome i in rep 1 with outcome i in rep 2)
        above diagonal: correlations between distinct outcomes in rep 1
        below diagonal: correlations between distinct outcome in rep 2
        '''
        ax.set_title(title)

        return fig

    def plot_guide_reproducibility(self, outcomes):
        rs = self.guide_r_series(outcomes)

        fig, ax = plt.subplots(figsize=(12, 6))

        sorted_rs = rs.sort_values(ascending=False)

        xs = np.arange(len(sorted_rs))

        df = pd.DataFrame(sorted_rs)
        df['x'] = xs
        df['color'] = 'black'
        df.loc[self.common_non_targeting_guides, 'color'] = 'tab:orange'

        kwargs = dict(x='x', y='r', linewidths=(0,), color='color')
        ax.scatter(data=df.query('color == "black"'), s=10, label='', **kwargs)
        ax.scatter(data=df.query('color != "black"'), s=25, label='non-targeting guides', **kwargs)

        ax.legend()

        ax.set_ylim(-0.75, 1)
        ax.set_xlim(-0.01 * len(rs), 1.01 * len(rs))

        ax.axhline(0, color='black')

        ax.set_xlabel('guides ranked by between-replicate correlation', size=12)
        ax.set_ylabel('correlation between replicates in outcome redistribution profile', size=12)

        return fig

    def compare_single_outcomes(self, num_outcomes=5, manual_data_lims=None, manual_ticks=None):
        x = self.pn0
        y = self.pn1

        outcomes = self.outcomes_above_simple_threshold(1e-3)

        log2_fold_changes = self.log2_fold_changes(outcomes, self.common_guides).stack().sort_index()

        fig, axs = plt.subplots(1, num_outcomes, figsize=(4 * num_outcomes, 2), gridspec_kw=dict(wspace=1))

        for outcome_i, ax in enumerate(axs):
            outcome = outcomes[outcome_i]
            df = log2_fold_changes.loc[outcome]

            if manual_data_lims is not None:
                data_lims = manual_data_lims
            else:
                data_lims = (np.floor(df.min().min() - 0.1), np.ceil(df.max().max() + 0.1))

            df['color'] = 'grey'

            df.loc[self.common_non_targeting_guides, 'color'] = 'C0'

            df = df.dropna()

            #enough_UMIs = pools[x].UMI_counts('perfect')['none'] >= 0

            r, p = scipy.stats.pearsonr(df[x], df[y])

            ax.scatter(x=x, y=y, data=df, color='color', linewidth=(0,), s=25, alpha=0.75)

            ax.set_xlim(*data_lims)
            ax.set_ylim(*data_lims)

            ax.set_xlabel(f'replicate 1')
            ax.set_ylabel(f'replicate 2')

            ax.set_title(f'{outcome_i}\n{outcome}')

            hits.visualize.draw_diagonal(ax, alpha=0.2)
            ax.axhline(0, color='black', alpha=0.2)
            ax.axvline(0, color='black', alpha=0.2)

            ax.annotate(f'r$^2$ = {r**2:0.2f}',
                        xy=(0, 1),
                        xycoords='axes fraction',
                        xytext=(10, -10),
                        textcoords='offset points',
                        ha='left',
                        va='top',
                        size=10,
                       )

            ax.set_aspect('equal')

            if manual_ticks is not None:
                ax.set_xticks(manual_ticks)
                ax.set_yticks(manual_ticks)

        return fig

    def compare_single_guides(self, num_guides=5, threshold=2e-3, manual_data_lims=None, manual_ticks=None):
        x = self.pn0
        y = self.pn1

        outcomes = self.outcomes_above_simple_threshold(threshold)

        log2_fold_changes = self.log2_fold_changes(outcomes, self.common_guides)

        r_series = self.guide_r_series(outcomes).sort_values(ascending=False)

        num_guides = 5

        fig, axs = plt.subplots(1, num_guides, figsize=(4 * num_guides, 2), gridspec_kw=dict(wspace=1))

        for guide_i, ax in enumerate(axs):
            guide = r_series.index[guide_i]
            
            df = log2_fold_changes.xs(guide, axis=1, level=1).copy()

            if manual_data_lims is not None:
                data_lims = manual_data_lims
            else:
                data_lims = (np.floor(df.min().min() - 0.1), np.ceil(df.max().max() + 0.1))

            df['color'] = 'black'

            #enough_UMIs = pools[x].UMI_counts('perfect')['none'] >= 0

            r, p = scipy.stats.pearsonr(df[x], df[y])
            #f = pools[x].non_targeting_fractions('perfect', 'none').loc[outcome]

            ax.scatter(x=x, y=y, data=df, color='color', linewidth=(0,), s=25, alpha=0.5, marker='D')

            ax.set_xlim(*data_lims)
            ax.set_ylim(*data_lims)

            ax.set_xlabel(f'replicate 1', size=12)
            ax.set_ylabel(f'replicate 2', size=12)

            ax.set_title(f'log2 fold-changes in\noutcome frequencies for {guide}')

            hits.visualize.draw_diagonal(ax, alpha=0.2)
            ax.axhline(0, color='black', alpha=0.2)
            ax.axvline(0, color='black', alpha=0.2)

            ax.annotate(f'r$^2$ = {r**2:0.2f}',
                        xy=(0, 1),
                        xycoords='axes fraction',
                        xytext=(10, -10),
                        textcoords='offset points',
                        ha='left',
                        va='top',
                        size=10,
                       )

            ax.set_aspect('equal')

            if manual_ticks is not None:
                ax.set_xticks(manual_ticks)
                ax.set_yticks(manual_ticks)

        return fig

    def active_guides_for_same_gene(self, threshold=5e-3, num_guides=100):
        outcomes = self.outcomes_above_simple_threshold(threshold)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        rs = self.guide_guide_correlations(threshold)

        for ax, pn in zip(axs, self.pn_pair):
            pool = self.pools[pn]
            active_guides = pool.top_n_active_guides(outcomes, num_guides)

            active_pairs = rs.query('guide_1 in @active_guides and guide_2 in @active_guides')
            same_gene = active_pairs.query('gene_1 == gene_2')

            bins = np.linspace(-1, 1, 30)
            common_kwargs = dict(bins=bins, linewidth=2, histtype='step', density=True)

            ax.hist(active_pairs[pn], **common_kwargs)
            ax.hist(same_gene[pn], **common_kwargs)
            ax.set_xlim(-1.02, 1.02)

        return fig

    def guide_guide_correlations(self, threshold=5e-3):
        outcomes = self.outcomes_above_simple_threshold(threshold)
        log2_fold_changes = self.log2_fold_changes(outcomes, self.common_guides)

        rs = {}
        for pn in self.pn_pair:
            all_corrs = log2_fold_changes[pn].corr().stack()

            rs[pn] = all_corrs

        rs = pd.DataFrame(rs)

        rs.index.names = ['guide_1', 'guide_2']

        # Make columns out of index levels.

        for i in [1, 2]:
            k = f'guide_{i}'
            rs[k] = rs.index.get_level_values(k)
            rs[f'gene_{i}'] = self.common_guides_df.loc[rs[k], 'gene'].values

        upper_triangle = rs.query('guide_1 < guide_2')

        return upper_triangle

    def guide_guide_correlation_scatter(self, threshold=5e-3, num_guides=100, bad_guides=None):
        outcomes = self.outcomes_above_simple_threshold(5e-3)
        guides = self.union_of_top_n_guides(outcomes, num_guides)
        if bad_guides is not None:
            guides = [g for g in guides if g not in bad_guides]

        rs = self.guide_guide_correlations(threshold)

        active_rs = rs.query('guide_1 in @guides and guide_2 in @guides').copy()

        active_rs['color'] = 'grey'

        active_rs.loc[active_rs['gene_1'] == active_rs['gene_2'], 'color'] = 'tab:red'

        g = sns.JointGrid(height=4)

        ax = g.ax_joint

        distinct = active_rs.query('color == "grey"')
        same = active_rs.query('color == "tab:red"')

        x = self.pn0
        y = self.pn1

        kwargs = dict(x=x, y=y, color='color', linewidths=(0,))
        ax.scatter(data=distinct, s=10, label='different target genes', alpha=0.3, **kwargs)
        ax.scatter(data=same, s=20, label='same target gene', alpha=0.8, **kwargs)

        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)

        ax.annotate('same target gene',
                    xy=(0, 1),
                    xycoords='axes fraction',
                    xytext=(3, -10),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    color='tab:red',
                    size=10,
                   )

        ax.annotate('different target genes',
                    xy=(0, 1),
                    xycoords='axes fraction',
                    xytext=(3, -23),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    color='grey',
                    size=10,
                   )

        hits.visualize.draw_diagonal(ax, alpha=0.2)
        ax.axhline(0, color='black', alpha=0.2)
        ax.axvline(0, color='black', alpha=0.2)

        ax.set_xlabel('correlation between\ndistinct CRISPRi guides,\nreplicate 1', size=12)
        ax.set_ylabel('correlation between\ndistinct CRISPRi guides,\nreplicate 2', size=12)

        for ax, pn, orientation in [(g.ax_marg_x, x, 'vertical'), (g.ax_marg_y, y, 'horizontal')]:
            bins = np.linspace(-1, 1, 30)
            kwargs = dict(histtype='step', bins=bins, density=True, orientation=orientation)

            max_n = 0
    
            n, *rest = ax.hist(distinct[pn], color='grey', **kwargs)
            max_n = max(max_n, max(n))
            
            n, *rest = ax.hist(same[pn], color='red', **kwargs)
            max_n = max(max_n, max(n))
            
            if orientation == 'vertical':
                set_lim = ax.set_ylim
            else:
                set_lim = ax.set_xlim
                
            set_lim(0, max_n * 1.05)

        return g.fig, active_rs

    def full_categories(self, outcomes):
        full_categories = defaultdict(list)

        for c, s, d in outcomes:
            ti = self.target_info
            if c == 'deletion':
                deletion = knock_knock.outcome.DeletionOutcome.from_string(d).undo_anchor_shift(ti.anchor)
                directionality = deletion.classify_directionality(ti)
                full_category = f'{c}, {directionality}'
            else:
                full_category = c
                
            full_categories[full_category].append((c, s, d))

        cat_color_order = [
            'SD-MMEJ',
            'deletion, PAM-distal',
            'deletion, PAM-proximal',
            'deletion, ambiguous',
            'deletion, bidirectional',
            'genomic insertion',
            'insertion',
            'wild type',
        ]

        palette = bokeh.palettes.Dark2[8]
        colors = dict(zip(cat_color_order, palette))

        category_aliases = {
            'wild type': 'unedited',
            'deletion, PAM-distal': 'deletion on PAM-distal side',
            'deletion, bidirectional': 'bidirectional deletion',
            'deletion, PAM-proximal': 'deletion on PAM-proximal side',
            'SD-MMEJ': 'insertion and deletion',
        }

        cat_display_order = [
            'wild type',
            'insertion',
            'deletion, bidirectional',
            'deletion, PAM-distal',
            'deletion, PAM-proximal',
            'deletion, ambiguous',
            'SD-MMEJ',
        ]

        return full_categories, colors, category_aliases, cat_display_order

    def get_bidirectional_deletions(self, threshold=5e-3):
        ti = self.pool0.target_info

        def get_min_removed(d):
            deletion = knock_knock.outcome.DeletionOutcome.from_string(d).undo_anchor_shift(ti.anchor)
            min_removed_before = max(0, ti.cut_after - max(deletion.deletion.starts_ats) + 1)
            min_removed_after = max(0, min(deletion.deletion.ends_ats) - ti.cut_after)
            return min_removed_before, min_removed_after

        relevant_outcomes = []

        for c, s, d in self.outcomes_above_simple_threshold(threshold):
            if c == 'deletion':
                min_before, min_after = get_min_removed(d)
                if min_before > 0 and min_after > 0:
                    relevant_outcomes.append((c, s, d))
                    
        return relevant_outcomes

class PoolReplicates:
    def __init__(self, pools, short_name):
        self.pools = pools
        self.short_name = short_name

        self.target_info = self.pools[0].target_info
        self.variable_guide_library = self.pools[0].variable_guide_library

    @memoized_property
    def outcome_fractions(self):
        return pd.concat({pool.group: pool.outcome_fractions('perfect')['none'] for pool in self.pools}, axis=1).fillna(0)
    
    @memoized_property
    def outcome_fraction_means(self):
        return self.outcome_fractions.mean(axis=1, level=1)

    @memoized_property
    def outcome_fraction_stds(self):
        return self.outcome_fractions.std(axis=1, level=1)

    @memoized_property
    def non_targeting_fraction_means(self):
        return self.outcome_fraction_means[ALL_NON_TARGETING].sort_values(ascending=False)

    @memoized_property
    def non_targeting_fraction_stds(self):
        return self.outcome_fraction_stds[ALL_NON_TARGETING].loc[self.non_targeting_fraction_means.index]

    @memoized_property
    def log2_fold_changes(self):
        fs = self.outcome_fraction_means
        fold_changes = fs.div(fs[ALL_NON_TARGETING], axis=0)
        return np.log2(fold_changes)

    @memoized_property
    def log2_fold_change_intervals(self):
        fs = self.outcome_fraction_means
        fold_changes = fs.div(fs[ALL_NON_TARGETING], axis=0)
        return np.log2(fold_changes)

    @memoized_property
    def category_fractions(self):
        all_fs = {pool.group: pool.category_fractions for pool in self.pools}
        return pd.concat(all_fs, axis=1).fillna(0)

    @memoized_property
    def category_fraction_means(self):
        return self.category_fractions.mean(axis=1, level=1)

    @memoized_property
    def category_fraction_stds(self):
        return self.category_fractions.std(axis=1, level=1)

    @memoized_property
    def categories_by_baseline_frequency(self):
        return self.category_fraction_means[ALL_NON_TARGETING].sort_values(ascending=False).index.values

    @memoized_property
    def category_fraction_differences(self):
        return pd.concat({pool.group: pool.category_fraction_differences for pool in self.pools}, axis=1).fillna(0)

    @memoized_property
    def category_fraction_difference_means(self):
        return self.category_fraction_differences.mean(axis=1, level=1)

    @memoized_property
    def category_fraction_difference_stds(self):
        return self.category_fraction_differences.std(axis=1, level=1)

    @memoized_property
    def category_log2_fold_changes(self):
        return pd.concat({pool.group: pool.category_log2_fold_changes for pool in self.pools}, axis=1).fillna(0)

    @memoized_property
    def category_log2_fold_change_means(self):
        return self.category_log2_fold_changes.mean(axis=1, level=1)

    @memoized_property
    def cateogry_log2_fold_change_stds(self):
        return self.category_log2_fold_changes.std(axis=1, level=1)

    @memoized_property
    def gene_level_category_statistics(self):
        return pd.concat({pool.group: pool.gene_level_category_statistics for pool in self.pools}, axis=1)

    @memoized_property
    def gene_level_category_statistic_means(self):
        return self.gene_level_category_statistics.mean(axis=1, level=[1, 2])

    def plot_ranked_category_statistics(self, category, stat='extreme_2', top_n=3, bottom_n=3, y_lim=(-2, 2)):
        df = self.gene_level_category_statistics.xs([category, stat], level=[1, 2], axis=1).copy()
        first, second = [pool.group for pool in self.pools]
        df['color'] = 'black'

        df.loc[['MSH2', 'MSH6'], 'color'] = 'tab:green'
        df.loc[['PMS2', 'MLH1'], 'color'] = 'tab:green'
        df.loc['HLTF', 'color'] = 'tab:orange'
        #df.loc[[f'POLD{i}' for i in range(1, 5)], 'color'] = 'tab:red'
        #df.loc[[f'RFC{i}' for i in range(1, 6)], 'color'] = 'tab:purple'
        #df.loc[['MRE11', 'RAD50', 'NBN'], 'color'] = 'tab:pink'

        df['mean'] = df.mean(axis=1).sort_values()

        df = df.sort_values('mean')
        df['x'] = np.arange(len(df))

        df = df.drop('negative_control', errors='ignore')
        
        fig, ax = plt.subplots(figsize=(6, 4))

        for to_plot, alpha in [
            (df.query('color == "black"'), 0.5),
            (df.query('color != "black"'), 0.9),
        ]:
            ax.scatter(x='x', y='mean', data=to_plot, linewidths=(0,), s=10, c='color', alpha=alpha, clip_on=False)

        for _, row in df.iterrows():
            xs = [row['x'], row['x']]
            ys = sorted(row[[first, second]])
            ax.plot(xs, ys, alpha=0.2, color=row['color'], clip_on=False)

        ax.axhline(0, color='black')
        ax.set_ylim(*y_lim)
        ax.set_xlim(-0.02 * len(df), 1.02 * len(df))

        label_kwargs = dict(
            ax=ax,
            xs='x',
            ys='mean',
            labels='gene', 
            initial_distance=10,
            arrow_alpha=0.15,
            color='color',
            avoid=True,
        )

        hits.visualize.label_scatter_plot(data=df.iloc[::-1][:top_n], vector='left', **label_kwargs)
        hits.visualize.label_scatter_plot(data=df.iloc[:bottom_n], vector='right', **label_kwargs)

        ax.set_xticks([])

        ax.set_ylabel('log2 fold-change from non-targeting\n(average of 2 most extreme guides,\naverage of 2 replicates)')

        for side in ['bottom', 'top', 'right']:
            ax.spines[side].set_visible(False)

        ax.grid(axis='y', alpha=0.2)
        
        ax.set_title(category, size=16)

        return fig