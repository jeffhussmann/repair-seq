import argparse
import gzip
import multiprocessing
import subprocess
import tempfile

from pathlib import Path
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import scanpy as sc 
import scipy.sparse
import yaml

from pomegranate import GeneralMixtureModel, PoissonDistribution, NormalDistribution

from hits import utilities, mapping_tools, fasta, fastq, bus, sam
from hits.utilities import memoized_property

import repair_seq.guide_library

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

def load_bustools_counts(prefix):
    prefix = str(prefix)
    data = sc.read_mtx(str(prefix) + '.mtx')
    data.obs.index = pd.read_csv(prefix + '.barcodes.txt', header=None)[0].values
    data.var.index = pd.read_csv(prefix + '.genes.txt', header=None)[0].values

    return data

def fit_mixture_model(counts):
    ''' Code adapted from https://github.com/josephreplogle/guide_calling '''

    data = np.log2(counts + 1)
    
    reshaped_data = data.reshape(-1, 1)
    
    xs = np.linspace(-2, max(data) + 2, 1000)

    # Re-fit the model until it has converged with both components given non-zero weight
    # and the Poisson component in the first position with lower mean.
    
    while True:
        model = GeneralMixtureModel.from_samples([PoissonDistribution, NormalDistribution], 2, reshaped_data)
        
        if 0 in model.weights:
            # One component was eliminated
            continue
        elif np.isnan(model.probability(xs)).any():
            continue
        elif model.distributions[0].parameters[0] > model.distributions[1].parameters[0]:
            continue
        elif model.distributions[0].name != 'PoissonDistribution':
            continue
        else:
            break
            
    labels = model.predict(reshaped_data)
    
    xs = np.linspace(0, max(data) + 2, 1000)
    p_second_component = model.predict_proba(xs.reshape(-1, 1))[:, 1]
    threshold = 2**xs[np.argmax(p_second_component >= 0.5)]
    
    return labels, threshold 

class PerturbseqLane:
    def __init__(self, base_dir, group, name):
        self.base_dir = Path(base_dir)
        self.group = group
        self.name = name

        self.data_dir = self.base_dir / 'data' / self.group

        self.barcode_length = 16
        self.UMI_length = 10

        full_sample_sheet = load_sample_sheet(self.data_dir / 'sample_sheet.yaml')

        sample_sheet = full_sample_sheet['lanes'][name]

        self.output_dir = self.base_dir / 'results' / self.group / self.name
        self.sgRNA_dir  = self.output_dir / 'sgRNA'
        self.GEX_dir  = self.output_dir / 'GEX'
        self.cellranger_dir = self.output_dir / 'cellranger_output'

        self.guide_library = repair_seq.guide_library.GuideLibrary(self.base_dir, full_sample_sheet['guide_libary'])
        self.guide_index = self.guide_library.fns['perturbseq_STAR_index']
        self.whitelist_fn = Path(full_sample_sheet['whitelist_fn'])

        self.sgRNA_fns = {
            'dir': self.sgRNA_dir,
            'R1_fns': [self.data_dir / fn for fn in sample_sheet['sgRNA_R1_fns']],
            'R2_fns': [self.data_dir / fn for fn in sample_sheet['sgRNA_R2_fns']],

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
            'R1_fns': [self.data_dir / fn for fn in sample_sheet['GEX_R1_fns']],
            'R2_fns': [self.data_dir / fn for fn in sample_sheet['GEX_R2_fns']],

            'bus': self.GEX_dir / 'output.bus',
            'counts': self.GEX_dir / 'counts',

            'kallisto_index': Path(full_sample_sheet['kallisto_index']),
            'genemap': Path(full_sample_sheet['kallisto_genemap']),
            'ecmap': self.GEX_dir / 'matrix.ec',
            'txnames': self.GEX_dir / 'transcripts.txt',

            'cellranger_filtered_feature_bc_matrix_dir': self.cellranger_dir / 'filtered_feature_bc_matrix',
            'cellranger_barcodes': self.cellranger_dir / 'filtered_feature_bc_matrix' / 'barcodes.tsv.gz',
            'sgRNA_counts_list': self.cellranger_dir / 'sgRNA_counts_list.csv.gz',
            'sgRNA_counts_csv': self.cellranger_dir / 'sgRNA_counts.csv.gz',
            'sgRNA_counts_h5ad': self.cellranger_dir / 'sgRNA_counts.h5ad',
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
            print(f'{self.name} specifies non-existent files: {[str(fn) for fn in missing_files]}')

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

    @memoized_property
    def sgRNA_data(self):
        return load_bustools_counts(self.sgRNA_fns['counts'])

    @memoized_property
    def GEX_data(self):
        return load_bustools_counts(self.GEX_fns['counts'])

    @memoized_property
    def ENSG_to_name(self):
        names_fn = '/lab/solexa_weissman/indices/refdata-cellranger-hg19-1.2.0/kallisto/transcripts_to_genes_hg19.txt'

        updated_names_fn = self.base_dir / 'guides' / 'DDR_library' / 'updated_gene_names.txt'
        updated_names = pd.read_csv(updated_names_fn, sep='\t', index_col='old_name', squeeze=True)
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

        ENSG_to_name['negative_control'] = 'negative_control'

        return pd.Series(ENSG_to_name)

    @memoized_property
    def name_to_ENSG(self):
        return pd.Series(utilities.reverse_dictionary(self.ENSG_to_name))

    def combine_sgRNA_and_GEX_counts(self):
        gex_data = self.GEX_data
        gex_data.var['name'] = [self.ENSG_to_name[g] for g in gex_data.var.index.values]

        sgRNA_data = self.sgRNA_data

        gex_data.obs['num_UMIs'] = np.sum(gex_data.X, axis=1).A1
        sgRNA_data.obs['num_UMIs'] = np.sum(sgRNA_data.X, axis=1).A1

        sgRNA_data.obs['highest_count'] = sgRNA_data.X.max(axis=1).todense().A1
        sgRNA_data.obs['highest_index'] = sgRNA_data.X.argmax(axis=1).A1
        sgRNA_data.obs['fraction_highest'] = sgRNA_data.obs['highest_count'] / sgRNA_data.obs['num_UMIs']

        gex_cellBCs = gex_data.obs_names
        gex_data.obs['sgRNA_highest_index'] = sgRNA_data.obs['highest_index'].reindex(gex_cellBCs, fill_value=-1).astype(int)
        gex_data.obs['sgRNA_highest_count'] = sgRNA_data.obs['highest_count'].reindex(gex_cellBCs, fill_value=0).astype(int)
        gex_data.obs['sgRNA_fraction_highest'] = sgRNA_data.obs['fraction_highest'].reindex(gex_cellBCs, fill_value=0)

        gex_data.obs['sgRNA_num_UMIs'] = sgRNA_data.obs['num_UMIs'].reindex(gex_cellBCs, fill_value=0).astype(int)
        gex_data.obs['sgRNA_name'] = [sgRNA_data.var_names[i] if i != -1 else 'none' for i in gex_data.obs['sgRNA_highest_index']]

        # For performance reasons, go ahead and discard any BCs with < 1000 UMIs.
        gex_data = gex_data[gex_data.obs.query('num_UMIs >= 1000').index]

        valid_cellBCs = gex_data.obs.query('num_UMIs > 5e3').index

        guide_calls, num_guides = self.fit_guide_count_mixture_models(valid_cellBCs)

        gex_data.obs['MM_guide_call'] = guide_calls.reindex(gex_cellBCs, fill_value='none')

        gex_data.obs['MM_num_guides'] = num_guides.reindex(gex_cellBCs, fill_value=-1).astype(int)

        gex_data.write(self.fns['annotated_counts'])

    def fit_guide_count_mixture_models(self, valid_cellBCs):
        sgRNA_data = self.sgRNA_data[valid_cellBCs]
        guides_present = np.zeros(sgRNA_data.X.shape)

        for g, guide in enumerate(sgRNA_data.var.index):
            labels, _ = fit_mixture_model(sgRNA_data.obs_vector(guide))
            
            guides_present[:, g] = labels

        guide_calls = sgRNA_data.var_names[guides_present.argmax(axis=1)].values
        num_guides = guides_present.sum(axis=1)

        guide_calls = pd.Series(guide_calls, index=sgRNA_data.obs_names)
        num_guides = pd.Series(num_guides, index=sgRNA_data.obs_names)

        return guide_calls, num_guides

    @memoized_property
    def cellranger_barcodes(self):
        bcs = []

        with gzip.open(self.GEX_fns['cellranger_barcodes'], 'rt') as fh:
            for line in fh:
                bcs.append(line.strip())

        return pd.Index(bcs)

    def make_guide_count_tables(self):
        cells_with_dummy_lane = [f'{cell_bc}-1' for cell_bc in self.sgRNA_data.obs_names]
        self.sgRNA_data.obs.index = cells_with_dummy_lane

        cells_in_both = self.cellranger_barcodes.intersection(self.sgRNA_data.obs_names)

        sgRNA_data = self.sgRNA_data[cells_in_both]
        sgRNA_data.write(self.GEX_fns['sgRNA_counts_h5ad'])

        df = sgRNA_data.to_df().astype(int)
        df.index.name = 'cell_barcode'
        df.columns.name = 'guide_identity'
        df.to_csv(self.GEX_fns['sgRNA_counts_csv'])

        stacked = df.stack()
        stacked.name = 'UMI_count'
        stacked.index.names = ('cell_barcode', 'guide_identity')
        stacked.to_csv(self.GEX_fns['sgRNA_counts_list'])

    @memoized_property
    def annotated_counts(self):
        return sc.read_h5ad(self.fns['annotated_counts'])

    def process(self):
        #self.map_sgRNA_reads()
        #self.convert_sgRNA_bam_to_bus()
        #self.convert_bus_to_counts(self.sgRNA_fns)

        #self.pseudoalign_GEX_reads()
        #self.convert_bus_to_counts(self.GEX_fns)

        #self.combine_sgRNA_and_GEX_counts()
        self.make_guide_count_tables()

def load_sample_sheet(sample_sheet_fn):
    sample_sheet = yaml.safe_load(Path(sample_sheet_fn).read_text())
    return sample_sheet

class MultipleLanes:
    def __init__(self, base_dir, group):
        self.base_dir = Path(base_dir)
        self.group = group

        sample_sheet_fn = self.base_dir / 'data' / group / 'sample_sheet.yaml'
        full_sample_sheet = load_sample_sheet(sample_sheet_fn)

        self.guide_library = repair_seq.guide_library.GuideLibrary(self.base_dir, full_sample_sheet['guide_libary'])

        self.lanes = [PerturbseqLane(self.base_dir, self.group, name) for name in full_sample_sheet['lanes']]

        self.results_dir = self.base_dir / 'results' / self.group
        self.cellranger_dir = self.results_dir / 'cellranger_aggregated' / 'outs'

        self.fns = {
            'all_cells': self.results_dir / 'all_cells.h5ad',
        }

        self.GEX_fns = {
            'cellranger_filtered_feature_bc_matrix_dir': self.cellranger_dir / 'filtered_feature_bc_matrix',
            'cellranger_barcodes': self.cellranger_dir / 'filtered_feature_bc_matrix' / 'barcodes.tsv.gz',
            'sgRNA_counts_list': self.cellranger_dir / 'sgRNA_counts_list.csv.gz',
            'sgRNA_counts_csv': self.cellranger_dir / 'sgRNA_counts.csv.gz',
            'sgRNA_counts_h5ad': self.cellranger_dir / 'sgRNA_counts.h5ad',
            'guide_assignments': self.cellranger_dir / 'guide_assignments.csv.gz',
        }

    @memoized_property
    def ENSG_to_name(self):
        return self.lanes[0].ENSG_to_name

    @memoized_property
    def name_to_ENSG(self):
        return self.lanes[0].name_to_ENSG

    def combine_counts(self):
        all_cells = {}
        for lane in self.lanes:
            gex_data = lane.annotated_counts
            #good_cell_query = 'num_UMIs > 0.5e4 and sgRNA_num_UMIs > 1e2 and sgRNA_highest_count / sgRNA_num_UMIs > 0.9'
            cells = gex_data.obs.index
            cells_with_lane = [f'{cell_bc}-{lane.name[-1]}' for cell_bc in cells]
            all_cells[lane.name] = gex_data
            all_cells[lane.name].obs.index = cells_with_lane
            
        all_Xs = scipy.sparse.vstack([all_cells[name].X for name in sorted(all_cells)])
        all_obs = pd.concat([all_cells[name].obs for name in sorted(all_cells)])
        all_var = all_cells[sorted(all_cells)[0]].var

        all_cells = sc.AnnData(all_Xs, all_obs, all_var)
        all_cells.write(self.fns['all_cells'])

    def make_guide_count_tables(self):
        all_sgRNA_counts = []
        for lane in self.lanes:
            sgRNA_counts = sc.read_h5ad(lane.GEX_fns['sgRNA_counts_h5ad'])
            lane_num = lane.name[-1]
            sgRNA_counts.obs.index = [f'{cell_bc.rsplit("-", 1)[0]}-{lane_num}' for cell_bc in sgRNA_counts.obs_names]
            all_sgRNA_counts.append(sgRNA_counts)

        sgRNA_data = ad.concat(all_sgRNA_counts)
        sgRNA_data.write(self.GEX_fns['sgRNA_counts_h5ad'])

        df = sgRNA_data.to_df().astype(int)
        df.index.name = 'cell_barcode'
        df.columns.name = 'guide_identity'
        df.to_csv(self.GEX_fns['sgRNA_counts_csv'])

        stacked = df.stack()
        stacked.name = 'UMI_count'
        stacked.index.names = ('cell_barcode', 'guide_identity')
        stacked.to_csv(self.GEX_fns['sgRNA_counts_list'])

    @memoized_property
    def cells(self):
        return sc.read_h5ad(self.fns['all_cells'])

def process_in_pool(lane):
    lane.process()

def parallel(lanes, max_procs):
    pool = multiprocessing.Pool(processes=max_procs)
    pool.map(process_in_pool, lanes.lanes)

    #lanes.combine_counts()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=Path, required=True)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--max_procs', type=int, required=True)

    args = parser.parse_args()

    lanes = MultipleLanes(args.base_dir, args.group)

    parallel(lanes, args.max_procs)
