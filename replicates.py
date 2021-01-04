import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats.stats import pearsonr
import seaborn as sns

import hits.utilities
memoized_property = hits.utilities.memoized_property

import knock_knock.outcome

import ddr.visualize
import ddr.visualize.clustermap
import ddr.visualize.outcome_diagrams

class ReplicatePair:
    def __init__(self, pools, pn_pair):
        self.pools = pools
        self.pn_pair = pn_pair
        self.pn0 = pn_pair[0]
        self.pn1 = pn_pair[1]

        self.pool0 = self.pools[pn_pair[0]]
        self.pool1 = self.pools[pn_pair[1]]

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

    def plot_outcome_diagrams_with_correlations(self, color, threshold=5e-3, n_guides=100, window=(-20, 20)):
        protospacer_color = hits.visualize.apply_alpha(color, 0.3)
        PAM_color = hits.visualize.apply_alpha(color, 0.9)
        cut_color = hits.visualize.apply_alpha(color, 0.9)

        outcomes = self.outcomes_above_simple_threshold(threshold)
        guides = self.union_of_top_n_guides(outcomes, n_guides)

        r_series = self.outcome_r_series(guides=guides)
        r_matrix, _ = self.outcome_r_matrix(outcomes, guides)

        g = ddr.visualize.outcome_diagrams.DiagramGrid(outcomes,
                                                       self.pools[self.pn0].target_info,
                                                       window=window,
                                                       protospacer_color=protospacer_color,
                                                       PAM_color=PAM_color,
                                                       cut_color=cut_color,
                                                       draw_wild_type_on_top=True,
                                                       flip_if_reverse=False,
                                                      )

        g.add_ax('frequency', width_multiple=10, title='percentage of outcomes\nfor non-targeting guides')
        g.plot_on_ax('frequency', self.average_nt_fractions, transform=lambda f: f * 100, marker='.', color=color, clip_on=False)

        #g.axs_by_name['frequency'].set_xlim(np.log10(0.49e-2), np.log10(40e-2))
        #g.style_log10_frequency_ax('frequency')

        g.add_ax('correlation', gap_multiple=2, width_multiple=10, title='correlation between replicates\nin log2 fold changes\n for active guides')
        g.plot_on_ax('correlation', r_series, marker='.', color=color, clip_on=False)
        g.axs_by_name['correlation'].set_xlim(0, 1)

        plt.setp(g.axs_by_name['frequency'].get_xticklabels(), size=8)
        plt.setp(g.axs_by_name['correlation'].get_xticklabels(), size=8)

        # Note reversing of row order.
        g.add_heatmap(r_matrix[::-1], 'r',
                      cmap=ddr.visualize.correlation_cmap,
                      draw_tick_labels=False,
                      vmin=-1, vmax=1,
                      gap_multiple=2,
                     )

        g.mark_subset(self.get_bidirectional_deletions(), color='tab:blue', title='bidirectional\ndeletions')

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

        ax.set_title('correlations between guides are reproducible')
        ax.set_ylabel('guide-guide correlation in replicate 2')
        ax.set_xlabel('guide-guide correlation in replicate 1')

        return fig

    def outcome_outcome_correlation_reproducibility(self, threshold=5e-3, n_guides=100):
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

        relevant_indices = [outcomes.index(outcome) for outcome in self.get_bidirectional_deletions()]

        df = rs.query('outcome_1 < outcome_2').copy()

        df['color'] = 'grey'
        df.loc[df['outcome_1'].isin(relevant_indices) & df['outcome_2'].isin(relevant_indices), 'color'] = 'tab:blue'

        x = self.pn0
        y = self.pn1

        fig, ax = plt.subplots(figsize=(6, 6))

        kwargs = dict(x=x, y=y, color='color', alpha=0.9, linewidths=(0,), s=60)
        ax.scatter(data=df.query('color == "grey"'), label='', **kwargs)
        ax.scatter(data=df.query('color != "grey"'), label='bidirectional deletions', **kwargs)

        ax.legend()

        kwargs = dict(color='black', alpha=0.3)
        ax.axhline(0, **kwargs)
        ax.axvline(0, **kwargs)
        ax.plot([-1, 1], [-1, 1], **kwargs)

        ax.set_title('correlations between outcomes are reproducible')
        ax.set_ylabel('outcome-outcome correlation in replicate 2')
        ax.set_xlabel('outcome-outcome correlation in replicate 1')

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

    def compare_single_outcomes(self, num_outcomes=5):
        x = self.pn0
        y = self.pn1

        outcomes = self.outcomes_above_simple_threshold(1e-3)

        log2_fold_changes = self.log2_fold_changes(outcomes, self.common_guides).stack().sort_index()

        fig, axs = plt.subplots(1, num_outcomes, figsize=(5 * num_outcomes, 5))

        for outcome_i, ax in enumerate(axs):
            outcome = outcomes[outcome_i]
            df = log2_fold_changes.loc[outcome]

            data_lims = (np.floor(df.min().min() - 0.1), np.ceil(df.max().max() + 0.1))

            df['color'] = 'grey'

            df.loc[self.common_non_targeting_guides, 'color'] = 'C0'

            df = df.dropna()

            #enough_UMIs = pools[x].UMI_counts('perfect')['none'] >= 0

            r, p = scipy.stats.pearsonr(df[x], df[y])

            ax.scatter(x=x, y=y, data=df, color='color', linewidth=(0,), s=25, alpha=0.8)

            ax.set_xlim(*data_lims)
            ax.set_ylim(*data_lims)

            ax.set_xlabel(f'replicate 1')
            ax.set_ylabel(f'replicate 2')

            ax.set_title(f'log2 fold-change from non-targeting\nin frequency of outcome {outcome_i}\n{outcome}')

            hits.visualize.draw_diagonal(ax, alpha=0.2)
            ax.axhline(0, color='black', alpha=0.2)
            ax.axvline(0, color='black', alpha=0.2)

            ax.annotate(f'$r^2 = {r**2:0.2f}$', xy=(0, 1), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', size=14)

            ax.set_aspect('equal')

        return fig

    def compare_single_guides(self, num_guides=5, threshold=2e-3, manual_data_lims=None):
        x = self.pn0
        y = self.pn1

        outcomes = self.outcomes_above_simple_threshold(threshold)

        log2_fold_changes = self.log2_fold_changes(outcomes, self.common_guides)

        r_series = self.guide_r_series(outcomes).sort_values(ascending=False)

        num_guides = 5

        fig, axs = plt.subplots(1, num_guides, figsize=(5 * num_guides, 5))

        for guide_i, ax in enumerate(axs):
            guide = r_series.index[guide_i]
            
            df = log2_fold_changes.xs(guide, axis=1, level=1).copy()

            if manual_data_lims is not None:
                data_lims = manual_data_lims
            else:
                data_lims = (np.floor(df.min().min() - 0.1), np.ceil(df.max().max() + 0.1))

            df['color'] = 'grey'

            #enough_UMIs = pools[x].UMI_counts('perfect')['none'] >= 0

            r, p = scipy.stats.pearsonr(df[x], df[y])
            #f = pools[x].non_targeting_fractions('perfect', 'none').loc[outcome]

            ax.scatter(x=x, y=y, data=df, color='color', linewidth=(0,), s=45, alpha=0.8)

            ax.set_xlim(*data_lims)
            ax.set_ylim(*data_lims)

            ax.set_xlabel(f'replicate 1')
            ax.set_ylabel(f'replicate 2')

            ax.set_title(f'log2 fold-changes in\noutcome frequencies for {guide}')

            hits.visualize.draw_diagonal(ax, alpha=0.2)
            ax.axhline(0, color='black', alpha=0.2)
            ax.axvline(0, color='black', alpha=0.2)

            ax.annotate(f'$\mathrm{{r}}^2 = {r**2:0.2f}$',
                        xy=(0, 1),
                        xycoords='axes fraction',
                        xytext=(10, -10),
                        textcoords='offset points',
                        ha='left',
                        va='top',
                        size=14,
                       )

            ax.set_aspect('equal')

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

    def guide_guide_correlation_scatter(self, threshold=5e-3, num_guides=100):
        outcomes = self.outcomes_above_simple_threshold(5e-3)
        guides = self.union_of_top_n_guides(outcomes, num_guides)

        rs = self.guide_guide_correlations(threshold)

        active_rs = rs.query('guide_1 in @guides and guide_2 in @guides').copy()

        active_rs['color'] = 'grey'

        active_rs.loc[active_rs['gene_1'] == active_rs['gene_2'], 'color'] = 'tab:red'

        g = sns.JointGrid()

        ax = g.ax_joint

        distinct = active_rs.query('color == "grey"')
        same = active_rs.query('color == "tab:red"')

        x = self.pn0
        y = self.pn1

        kwargs = dict(x=x, y=y, color='color', alpha=0.8, linewidths=(0,))
        ax.scatter(data=same, s=20, label='guides for same gene', **kwargs)
        ax.scatter(data=distinct, s=10, label='guides for distinct genes', **kwargs)

        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)

        ax.legend()
        hits.visualize.draw_diagonal(ax, alpha=0.2)
        ax.axhline(0, color='black', alpha=0.2)
        ax.axvline(0, color='black', alpha=0.2)

        ax.set_xlabel('guide-guide correlation in replicate 1', size=12)
        ax.set_ylabel('guide-guide correlation in replicate 2', size=12)

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

        return g.fig

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
