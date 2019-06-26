import tempfile
import subprocess
import argparse
import multiprocessing

from pathlib import Path
from collections import Counter

import yaml
import pandas as pd
import numpy as np
import pysam
import scanpy as sc 

from hits import utilities, mapping_tools, fasta, fastq, bus, sam

def build_guide_index(guides_fn, index_dir):
    ''' index entries are in same orientation as R2 '''
    index_dir = Path(index_dir)
    index_dir.mkdir(exist_ok=True)

    fasta_fn = index_dir / 'expected_R2s.fasta'

    guides_df = pd.read_csv(guides_fn, sep='\t', index_col=0)

    before_ps = 'AGTACCAAGTTGATAACGGACTAGCCTTATTTAAACTTGCTATGCTGTTTCCAGCTTAGCTCTTAAAC'
    # Note: Cs here are from untemplated addition and are not deterministically 3.
    after_ps = 'CCCATATAAGAAA'

    with fasta_fn.open('w') as fh:
        for name, protospacer in guides_df['protospacer'].items():
            expected_R2 = before_ps + utilities.reverse_complement(protospacer) + after_ps
            fh.write(str(fasta.Record(name, expected_R2)))

    pysam.faidx(str(fasta_fn))

    mapping_tools.build_STAR_index([fasta_fn], index_dir)

    bustools_dir = index_dir / 'bustools_annotations'
    bustools_dir.mkdir(exist_ok=True)

    matrix_fn = bustools_dir / 'matrix.ec'
    with matrix_fn.open('w') as fh:
        for i, name in enumerate(guides_df.index):
            fh.write(f'{i}\t{i}\n')

    transcript_to_gene_fn = bustools_dir / 'transcripts_to_genes.txt'
    with transcript_to_gene_fn.open('w') as fh:
        for i, name in enumerate(guides_df.index):
            fh.write(f'{name}\t{name}\t{name}\n')
        
    transcripts_fn = bustools_dir / 'transcripts.txt'
    with transcripts_fn.open('w') as fh:
        for i, name in enumerate(guides_df.index):
            fh.write(f'{name}\n')

class PerturbseqLane():
    def __init__(self, full_sample_sheet, name):
        self.name = name
        self.barcode_length = 16
        self.UMI_length = 10

        sample_sheet = full_sample_sheet[name]

        self.output_dir = Path(sample_sheet['output_dir'])
        self.sgRNA_dir  = self.output_dir / 'sgRNA'
        self.GEX_dir  = self.output_dir / 'GEX'

        self.guide_index = Path(sample_sheet['guide_index'])
        self.whitelist_fn = Path(sample_sheet['whitelist_fn'])

        self.sgRNA_fns = {
            'dir': self.sgRNA_dir,
            'R1_fns': [Path(fn) for fn in sample_sheet['sgRNA_R1_fns']],
            'R2_fns': [Path(fn) for fn in sample_sheet['sgRNA_R2_fns']],

            'STAR_output_prefix': self.sgRNA_dir / 'STAR' / 'sgRNA.',
            'bam': self.sgRNA_dir / 'sgRNA.bam',
            'bus': self.sgRNA_dir / 'sgRNA.bus',
            'counts': self.sgRNA_dir / 'counts',

            'genemap': self.guide_index / 'bustools_annotations' / 'transcripts_to_genes.txt',
            'ecmap': self.guide_index / 'bustools_annotations' / 'matrix.ec',
            'txnames': self.guide_index / 'bustools_annotations' / 'transcripts.txt',
        }
        
        self.GEX_fns = {
            'dir': self.GEX_dir,
            'R1_fns': [Path(fn) for fn in sample_sheet['GEX_R1_fns']],
            'R2_fns': [Path(fn) for fn in sample_sheet['GEX_R2_fns']],

            'bus': self.GEX_dir / 'output.bus',
            'counts': self.GEX_dir / 'counts',

            'kallisto_index': Path(sample_sheet['kallisto_index']),
            'genemap': Path(sample_sheet['kallisto_genemap']),
            'ecmap': self.GEX_dir / 'matrix.ec',
            'txnames': self.GEX_dir / 'transcripts.txt',
        }

        self.fns = {
            'annotated_counts': self.output_dir / 'counts.h5ad',
        }

        missing_files = []
        files_to_check = [
            self.sgRNA_fns['R1_fns'],
            self.sgRNA_fns['R2_fns'],
            self.GEX_fns['R1_fns'],
            self.GEX_fns['R2_fns'],
        ]
        for fns in files_to_check:
            for fn in fns:
                if not fn.exists():
                    missing_files.append(fn)

        if missing_files:
            raise ValueError(f'{self.name} specifies non-existent files: {[str(fn) for fn in missing_files]}')

    def map_sgRNA_reads(self):
        output_prefix = self.sgRNA_fns['STAR_output_prefix']
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        mapping_tools.map_STAR(self.sgRNA_fns['R2_fns'],
                               self.guide_index,
                               output_prefix,
                               mode='guide_alignment',
                               sort=False,
                               include_unmapped=True,
                               num_threads=1,
                               bam_fn=self.sgRNA_fns['bam'],
                              )

    def convert_sgRNA_bam_to_bus(self):
        barcode_length = self.barcode_length
        UMI_length = self.UMI_length

        R1s = fastq.reads(self.sgRNA_fns['R1_fns'], up_to_space=True)
        R2_alignment_groups = sam.grouped_by_name(self.sgRNA_fns['bam'])

        with self.sgRNA_fns['bus'].open('wb') as fh:
            bus.write_header(fh, self.barcode_length, self.UMI_length)

            for (qname, als), R1 in zip(R2_alignment_groups, R1s):
                if qname != R1.name:
                    raise ValueError(qname, R1.name)

                ref_ids = {al.reference_id for al in als if not al.is_unmapped}
                if len(ref_ids) == 1:
                    barcode = R1.seq[:self.barcode_length]
                    UMI = R1.seq[barcode_length:barcode_length + UMI_length]
                    if 'N' in barcode or 'N' in UMI:
                        continue
                    record = bus.Record(barcode, UMI, ref_ids.pop(), 1, 0)
                    fh.write(record.pack())

    def convert_bus_to_counts(self, fns):
        with tempfile.TemporaryDirectory(prefix=str(fns['dir'] / 'tmp')) as temp_dir_name:
            correct_command = [
                'bustools', 'correct',
                '--whitelist', str(self.whitelist_fn),
                '--pipe',
                str(fns['bus']),
            ]
            correct_process = subprocess.Popen(correct_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            sort_command = [
                'bustools', 'sort',
                '--temp', temp_dir_name,
                '--pipe',
                '-',
            ]
            sort_process = subprocess.Popen(sort_command, stdin=correct_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            correct_process.stdout.close()

            count_command = [
                'bustools', 'count',
                '--output', str(fns['counts']),
                '--genemap', str(fns['genemap']),
                '--ecmap', str(fns['ecmap']),
                '--txnames', str(fns['txnames']),
                '--genecounts',
                '-',
            ]
            count_process = subprocess.Popen(count_command, stdin=sort_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            sort_process.stdout.close()
            count_process.communicate()

    def pseudoalign_GEX_reads(self):
        self.GEX_dir.mkdir(parents=True, exist_ok=True)
        fastq_fns = []
        for R1_fn, R2_fn in zip(self.GEX_fns['R1_fns'], self.GEX_fns['R2_fns']):
            fastq_fns.extend([str(R1_fn), str(R2_fn)])

        kallisto_command = [
            'kallisto', 'bus',
            '-i', str(self.GEX_fns['kallisto_index']),
            '-x', '10xv2',
            '-o', str(self.GEX_dir),
        ]
        kallisto_command.extend(fastq_fns)

        try:
            subprocess.run(kallisto_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            for line in e.stdout.splitlines():
                print(line.decode())
            for line in e.stderr.splitlines():
                print(line.decode())
            raise

    def combine_sgRNA_and_GEX_counts(self):
        names_fn = '/nvme/indices/refdata-cellranger-hg19-1.2.0/kallisto/transcripts_to_genes_hg19.txt'

        updated_names = pd.read_csv('/home/jah/projects/ddr/guides/DDR_library/updated_gene_names.txt', sep='\t', index_col='old_name', squeeze=True)
        ENSG_to_name = {}

        names_seen = Counter()

        for line in open(names_fn):
            ENST, ENSG, name = line.strip().split()
            
            if ENSG in ENSG_to_name:
                continue
            
            names_seen[name] += 1
            
            if name in updated_names:
                name = updated_names[name]
            
            if names_seen[name] > 1:
                name_to_use = f'{name}_{names_seen[name]}'
            else:
                name_to_use = name
                
            ENSG_to_name[ENSG] = name_to_use

        def load_bustools_counts(prefix):
            prefix = str(prefix)
            data = sc.read_mtx(str(prefix) + '.mtx')
            data.obs.index = pd.read_csv(prefix + '.barcodes.txt', header=None)[0].values
            data.var.index = pd.read_csv(prefix + '.genes.txt', header=None)[0].values
            
            return data

        gex_data = load_bustools_counts(self.GEX_fns['counts'])
        gex_data.var['name'] = [ENSG_to_name[g] for g in gex_data.var.index.values]

        sgRNA_data = load_bustools_counts(self.sgRNA_fns['counts'])

        gex_data.obs['num_UMIs'] = np.sum(gex_data.X, axis=1).A1
        sgRNA_data.obs['num_UMIs'] = np.sum(sgRNA_data.X, axis=1).A1

        sgRNA_data.obs['highest_count'] = sgRNA_data.X.max(axis=1).todense().A1
        sgRNA_data.obs['highest_index'] = sgRNA_data.X.argmax(axis=1).A1
        sgRNA_data.obs['fraction_highest'] = sgRNA_data.obs['highest_count'] / sgRNA_data.obs['num_UMIs']

        gex_data.obs['sgRNA_highest_index'] = sgRNA_data.obs['highest_index'][gex_data.obs.index.values].fillna(-1).astype(int)
        gex_data.obs['sgRNA_highest_count'] = sgRNA_data.obs['highest_count'][gex_data.obs.index.values].fillna(0).astype(int)
        gex_data.obs['sgRNA_fraction_highest'] = sgRNA_data.obs['fraction_highest'][gex_data.obs.index.values].fillna(0)

        gex_data.obs['sgRNA_num_UMIs'] = sgRNA_data.obs['num_UMIs'][gex_data.obs.index.values]
        gex_data.obs['sgRNA_name'] = [sgRNA_data.var_names[i] if i != -1 else 'none' for i in gex_data.obs['sgRNA_highest_index']]

        gex_data.write(self.fns['annotated_counts'])

    def process(self):
        #self.map_sgRNA_reads()
        #self.convert_sgRNA_bam_to_bus()
        #self.convert_bus_to_counts(self.sgRNA_fns)

        #self.pseudoalign_GEX_reads()
        #self.convert_bus_to_counts(self.GEX_fns)

        self.combine_sgRNA_and_GEX_counts()

def process_in_pool(lane):
    lane.process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_sheet', type=Path, required=True)
    parser.add_argument('--max_procs', type=int, required=True)

    args = parser.parse_args()
    sample_sheet = yaml.safe_load(args.sample_sheet.read_text())

    lanes = [PerturbseqLane(sample_sheet, name) for name in sample_sheet]
    pool = multiprocessing.Pool(processes=args.max_procs)
    pool.map(process_in_pool, lanes)
