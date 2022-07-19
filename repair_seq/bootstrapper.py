from collections import defaultdict

import numpy as np
import pandas as pd
import hits.utilities

memoized_with_key = hits.utilities.memoized_with_key
memoized_property = hits.utilities.memoized_property

class Bootstrapper():
    def __init__(self, all_cells, guides):
        self.all_cells = all_cells.copy()
        self.guides = guides
        self._guide_cell_bcs = {}

        self.guide_UMI_fraction_bootstraps = defaultdict(list)
    
    def compute_guide_UMI_fractions(self, guide, bootstrap=False):
        cell_bcs = self.guide_cell_bcs(guide)
        if bootstrap:
            cell_bcs = np.random.choice(cell_bcs, size=len(cell_bcs), replace=True)
            
        cells = self.all_cells[cell_bcs]
        UMI_counts_per_gene = cells.X.A.sum(axis=0)
        total_UMIs = cells.obs['num_UMIs'].sum()
        UMI_fractions_per_gene = UMI_counts_per_gene / total_UMIs
        
        return UMI_fractions_per_gene
          
    @memoized_with_key
    def guide_UMI_fractions(self, guide):
        return self.compute_guide_UMI_fractions(guide)
        
    @memoized_with_key
    def guide_cell_bcs(self, guide):
        if guide == 'non-targeting':
            nt_guides = [g for g in guides if g.startswith('non-')]
            cell_bcs = self.all_cells.obs[self.all_cells.obs['sgRNA_name'].isin(nt_guides)].index
        else:
            cell_bcs = self.all_cells.obs.query('sgRNA_name == @guide').index

        return cell_bcs
                
    @memoized_property
    def guide_UMI_fractions_df(self):
        fs = [self.guide_UMI_fractions(guide) for guide in self.guides]
        return pd.DataFrame(fs, index=self.guides, columns=[ENSG_to_name[ensg] for ensg in ENSGs])
    
    @memoized_property
    def guide_knockdown_df(self):
        return self.guide_UMI_fractions_df / self.guide_UMI_fractions('non-targeting')
    
    def quantile_UMI_fractions_df(self, q):
        fs = [self.quantile_guide_UMI_fractions(guide, q) for guide in self.guides]
        return pd.DataFrame(fs, index=self.guides, columns=[ENSG_to_name[ensg] for ensg in ENSGs])
    
    def quantile_knockdown_df(self, q):
        return self.quantile_UMI_fractions_df(q) / self.guide_UMI_fractions('non-targeting')
        
    def add_bootstrap(self, guide):
        guide_fs = self.compute_guide_UMI_fractions(guide, bootstrap=True)
        self.guide_UMI_fraction_bootstraps[guide].append(guide_fs)
        
    def add_all_bootstraps(self, n):
        for guide in progress(self.guides):
            for i in range(n):
                self.add_bootstrap(guide)
        
    def quantile_guide_UMI_fractions(self, guide, q):
        bootstraps = np.array(self.guide_UMI_fraction_bootstraps[guide])
        return np.quantile(bootstraps, q, axis=0)
    
    @memoized_property
    def guide_knockdown(self):
        kds = []
        targeting_guides = []
        for guide in self.guides:
            gene = guides_df.loc[guide, 'gene']
            if gene != 'negative_control':
                kd = self.guide_knockdown_df.loc[guide, gene]
                kds.append(kd)
                targeting_guides.append(guide)
            
        return pd.Series(kds, index=targeting_guides)