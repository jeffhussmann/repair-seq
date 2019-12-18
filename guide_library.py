from pathlib import Path

import pandas as pd

from hits import utilities, fasta, mapping_tools

memoized_property = utilities.memoized_property

class GuideLibrary():
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

            'protospacers': self.full_dir / 'protospacers.fasta',
            'protospacer_mappings_dir': self.full_dir / 'protospacers',
            'protospacer_mappings_STAR_prefix': self.full_dir / 'protospacers' / 'mappings.',
        }

    @memoized_property
    def guides_df(self):
        guides_df = pd.read_table(self.fns['guides'], index_col='short_name')

        if 'promoter' in guides_df.columns:
            guides_df.loc[guides_df['promoter'].isnull(), 'promoter'] = 'P1P2'
        else:
            guides_df['promoter'] = 'P1P2'

        guides_df['best_promoter'] = True

        for gene, promoter in self.best_promoters.items():
            not_best = guides_df.query('gene == @gene and promoter != @promoter').index
            guides_df.loc[not_best, 'best_promoter'] = False

        return guides_df
    
    @memoized_property
    def best_promoters(self):
        if self.fns['best_promoters'].exists():
            best_promoters = pd.read_table(self.fns['best_promoters'], index_col='gene', squeeze=True)
        else:
            best_promoters = {}

        return best_promoters

    @memoized_property
    def old_gene_to_new_gene(self):
        updated_gene_names = pd.read_table(self.fns['updated_gene_names'], index_col=0, squeeze=True)
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

    def map_protospacers(self):
        self.fns['protospacer_mappings_dir'].mkdir(exist_ok=True)

        mapping_tools.map_STAR(self.fns['protospacers'],
                               self.reference_STAR_index,
                               self.fns['protospacer_mappings_STAR_prefix'],
                               mode='guide_alignment',
                              )
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

    def gene_guides(self, gene, only_best_promoter=False):
        query = 'gene == @gene'
        if only_best_promoter:
            query += ' and best_promoter'

        gene_guides = self.guides_df.query(query).sort_values(['gene', 'promoter', 'rank'])

        return gene_guides.index

    def guide_to_gene(self, guide):
        return self.guides_df.loc[guide]['gene']

    @memoized_property
    def guide_barcodes(self):
        return self.guides_df['guide_barcode']