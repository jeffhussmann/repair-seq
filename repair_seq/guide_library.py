from pathlib import Path
from collections import defaultdict

import pandas as pd

from hits import utilities, fasta, mapping_tools

memoized_property = utilities.memoized_property
memoized_with_args = utilities.memoized_with_args

class GuideLibrary:
    def __init__(self, base_dir, name):
        self.base_dir = Path(base_dir)
        self.name = name

        self.full_dir = self.base_dir / 'guides' / name

        self.reference_STAR_index = '/nvme/indices/refdata-cellranger-GRCh38-1.2.0/star'

        self.fns = {
            'guides': self.full_dir / 'guides.txt',
            'guides_fasta': self.full_dir / 'guides.fasta',
            'best_promoters': self.full_dir / 'best_promoters.txt',
            'updated_gene_names': self.full_dir / 'updated_gene_names.txt',

            'non_targeting_guide_sets': self.full_dir / 'non_targeting_guide_sets.txt',

            'protospacers': self.full_dir / 'protospacers.fasta',
            'perturbseq_STAR_index': self.full_dir / 'perturbseq_STAR_index',

            'cell_cycle_phase_fractions': self.full_dir / 'cell_cycle_phase_fractions.txt',
            'cell_cycle_log2_fold_changes': self.full_dir / 'cell_cycle_effects.txt',
            
            'K562_knockdown': self.full_dir / 'K562_knockdown.txt',
        }

    @memoized_property
    def guides_df(self):
        guides_df = pd.read_csv(self.fns['guides'], index_col='short_name', sep='\t')

        if 'promoter' in guides_df.columns:
            guides_df.loc[guides_df['promoter'].isnull(), 'promoter'] = 'P1P2'
        else:
            guides_df['promoter'] = 'P1P2'

        guides_df['best_promoter'] = True

        for gene, promoter in self.best_promoters.items():
            not_best = guides_df.query('gene == @gene and promoter != @promoter').index
            guides_df.loc[not_best, 'best_promoter'] = False

        guides_df = guides_df.sort_values(['gene', 'promoter', 'rank'])

        return guides_df
    
    @memoized_property
    def best_promoters(self):
        if self.fns['best_promoters'].exists():
            best_promoters = pd.read_csv(self.fns['best_promoters'], index_col='gene', squeeze=True, sep='\t')
        else:
            best_promoters = {}

        return best_promoters

    @memoized_property
    def old_gene_to_new_gene(self):
        updated_gene_names = pd.read_csv(self.fns['updated_gene_names'], index_col=0, squeeze=True, seq='\t')
        return updated_gene_names
    
    @memoized_property
    def new_gene_to_old_gene(self):
        new_to_old_dict = utilities.reverse_dictionary(self.old_gene_to_new_gene)
        def new_gene_to_old_gene(new_gene):
            return new_to_old_dict.get(new_gene, new_gene)

        return new_gene_to_old_gene
    
    def make_protospacer_fasta(self):
        with open(self.fns['protospacers'], 'w') as fh:
            for name, seq in self.guides_df['protospacer'].items():
                # Remove initial G from seq.
                record = fasta.Record(name, seq[1:])
                fh.write(str(record))

    def make_guides_fasta(self):
        with open(self.fns['guides_fasta'], 'w') as fh:
            for name, seq in self.guides_df['full_seq'].items():
                record = fasta.Record(name, seq)
                fh.write(str(record))

    @memoized_property
    def guides(self):
        guides = self.guides_df.index.values
        return guides

    @memoized_property
    def non_targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' in g]

    @memoized_property
    def targeting_guides(self):
        return [g for g in self.guides if 'non-targeting' not in g and 'eGFP' not in g]

    @memoized_property
    def genes(self):
        return sorted(set(self.guides_df['gene']))

    @memoized_property
    def genes_with_non_targeting_guide_sets(self):
        genes = self.genes + sorted(self.non_targeting_guide_sets)
        return genes

    def gene_guides(self, gene, only_best_promoter=False):
        if isinstance(gene, str):
            genes = [gene]
        else:
            genes = gene

        query = 'gene in @genes'
        if only_best_promoter:
            query += ' and best_promoter'

        gene_guides = self.guides_df.query(query).sort_values(['gene', 'promoter', 'rank'])

        nt_guides = []
        for gene in genes:
            nt_guides.extend(self.non_targeting_guide_sets.get(gene, []))

        all_guides = list(gene_guides.index) + nt_guides

        return all_guides

    @memoized_property
    def guide_to_gene(self):
        guide_to_gene = self.guides_df['gene'].copy()
        return guide_to_gene

    @memoized_property
    def guide_to_gene_with_non_targeting_guide_sets(self):
        guide_to_gene = self.guides_df['gene'].copy()
        for nt_guide_set, guides in self.non_targeting_guide_sets.items():
            for guide in guides:
                guide_to_gene[guide] = nt_guide_set
        return guide_to_gene

    @memoized_property
    def guide_barcodes(self):
        return self.guides_df['guide_barcode']

    @memoized_with_args
    def gene_indices(self, gene):
        idxs = [i for i, g in enumerate(self.guides_df['gene']) if g == gene]
        return min(idxs), max(idxs)

    @memoized_property
    def cell_cycle_log2_fold_changes(self):
        return pd.read_csv(self.fns['cell_cycle_log2_fold_changes'], index_col=0)

    @memoized_property
    def cell_cycle_phase_fractions(self):
        return pd.read_csv(self.fns['cell_cycle_phase_fractions'], index_col=0)

    @memoized_property
    def K562_knockdown(self):
        return pd.read_csv(self.fns['K562_knockdown'], index_col=0)

    @memoized_property
    def non_targeting_guide_sets(self):
        non_targeting_guide_sets = {}

        with open(self.fns['non_targeting_guide_sets']) as fh:
            for i, line in enumerate(fh):
                guides = line.strip().split(',')

                set_name = f'non-targeting_set_{i:05d}'

                non_targeting_guide_sets[set_name] = list(guides)

        return non_targeting_guide_sets

class DummyGuideLibrary:
    def __init__(self):
        self.guides = ['none']
        self.non_targeting_guides = ['none']
        self.genes = ['negative_control']
        self.name = None
    
    @memoized_property
    def guide_to_gene(self):
        return defaultdict(lambda: 'none')

    def gene_guides(self, gene, **kwargs):
        if gene == 'negative_control':
            return ['none']
        else:
            return []

dummy_guide_library = DummyGuideLibrary()
