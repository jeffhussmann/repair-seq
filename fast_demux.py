import gzip
import itertools
import multiprocessing
from pathlib import Path
from collections import defaultdict, deque

import yaml
import tqdm

from hits import fastq, fasta, utilities
import ddr.collapse

def chunk_number_to_string(chunk_number):
    return f'{chunk_number:06d}'

class FastqChunker():
    def __init__(self, base_dir, quartet_name, which, queue=None, reads_per_chunk=int(5e6)):
        self.base_dir = Path(base_dir)
        self.quartet_name = quartet_name
        self.which = which
        self.reads_per_chunk = reads_per_chunk
        self.queue = queue
        
        sample_sheet_fn = self.base_dir  / 'sample_sheet.yaml'
        sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
        quartet = sample_sheet['quartets'][quartet_name]
        
        self.fn_name = quartet[self.which]
        
        self.input_fn = (self.base_dir / self.fn_name).with_suffix('.fastq.gz')
        
        self.chunks_dir = self.base_dir / 'chunks'
        self.chunks_dir.mkdir(exist_ok=True)
        
        self.current_chunk_number = 0
        
        self.current_fh = None
        
    def close(self):
        if self.current_fh is not None:
            self.current_fh.close()
            if self.queue is not None:
                self.queue.put(('chunk', self.quartet_name, self.which, self.current_chunk_number))
                
            self.current_chunk_number += 1
            
    def get_chunk_fn(self, chunk_number):
        chunk_string = chunk_number_to_string(chunk_number)
        next_fn = self.chunks_dir / f'{self.fn_name}_{chunk_string}.fastq.gz'
        return next_fn
            
    def start_next_chunk(self):
        self.close()
        
        chunk_fn = self.get_chunk_fn(self.current_chunk_number)
        self.current_fh = gzip.open(str(chunk_fn), 'wt', compresslevel=1)
        
    def split_into_chunks(self):
        line_groups = itertools.islice(fastq.get_line_groups(self.input_fn), int(5e7))
        
        for read_number, line_group in enumerate(line_groups):
            if read_number % self.reads_per_chunk == 0:
                self.start_next_chunk()
            
            for line in line_group:
                self.current_fh.write(line)
                
        self.close()
        
        if self.queue is not None:
            self.queue.put(('chunk', self.quartet_name, self.which, 'DONE'))
                
    def get_all_chunk_numbers(self):
        numbers = []
        for fn in sorted(self.chunks_dir.glob(f'{self.fn_name}_*.fastq.gz')):
            numbers.append(int(fn.name[len(f'{self.fn_name}_'):-len('.fastq.gz')]))
        return numbers
    
class UMISorters():
    def __init__(self, base_dir, quartet_name, chunk_number):
        self.base_dir = Path(base_dir)
        
        sample_sheet_fn = self.base_dir  / 'sample_sheet.yaml'
        sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
        self.group_name = sample_sheet['group_name']
        
        self.chunk_string = f'{quartet_name}_{chunk_number_to_string(chunk_number)}'

        self.sorters = defaultdict(list)

    def __enter__(self):
        return self

    def __getitem__(self, key):
        return self.sorters[key]

    def __exit__(self, exception_type, exception_value, exception_traceback):
        for sample, fixed_guide, variable_guide in sorted(self.sorters):
            reads = self.sorters[sample, fixed_guide, variable_guide]
            sorted_reads = sorted(reads, key=lambda r: r.name)

            output_dir = self.base_dir.parent / f'{self.group_name}_{sample}' / 'by_guide' / self.chunk_string
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fn = output_dir / f'{fixed_guide}-{variable_guide}_R2.fastq.gz'
            
            with gzip.open(str(fn), 'wt', compresslevel=1) as zfh:
                for read in sorted_reads:
                    zfh.write(str(read))

            del self.sorters[sample, fixed_guide, variable_guide]
            del sorted_reads
            
def split_into_chunks(base_dir, quartet_name, which, queue=None):
    chunker = FastqChunker(base_dir, quartet_name, which, queue=queue)
    chunker.split_into_chunks()
    
def demux_chunk(base_dir, quartet_name, chunk_number, queue):
    sample_sheet_fn = Path(base_dir)  / 'sample_sheet.yaml'
    sample_sheet = yaml.safe_load(sample_sheet_fn.read_text())
    sample_indices = sample_sheet['sample_indices']
    sample_index_resolver = utilities.get_one_mismatch_resolver(sample_indices)

    guide_barcodes = {'none': 'TGCACGTA'}
    fixed_guide_barcode_resolver = utilities.get_one_mismatch_resolver(guide_barcodes)
    R2_guide_barcode_slice = slice(6, 14)

    guides = fasta.to_dict('/home/jah/projects/ddr/guides/DDR_sublibrary/guides.fasta')
    variable_guide_resolver = utilities.get_one_mismatch_resolver({g: s[:45] for g, s in guides.items()})
    
    Annotation = ddr.collapse.Annotations['UMI_guide']
    
    fastq_fns = [FastqChunker(base_dir, quartet_name, which).get_chunk_fn(chunk_number) for which in fastq.quartet_order]

    sorters = UMISorters(base_dir, quartet_name, chunk_number)
    with sorters:
        for quartet in fastq.read_quartets(fastq_fns, up_to_space=True):
            if len({r.name for r in quartet}) != 1:
                raise ValueError('quartet out of sync')

            sample = sample_index_resolver.get(quartet.I2.seq, 'unknown')

            variable_guide = variable_guide_resolver.get(quartet.R1.seq, 'unknown')

            guide_barcode = quartet.R2.seq[R2_guide_barcode_slice]
            fixed_guide = fixed_guide_barcode_resolver.get(guide_barcode, 'unknown')

            if 'unknown' in {sample, variable_guide, fixed_guide}:
                continue

            original_name = quartet.R1.name
            UMI = quartet.I1.seq
            guide_seq = quartet.R1.seq
            guide_qual = fastq.sanitize_qual(quartet.R1.qual)

            annotation = Annotation(guide=guide_seq,
                                    guide_qual=guide_qual,
                                    original_name=original_name,
                                    UMI=UMI,
                                   )
            quartet.R2.name = str(annotation)

            sorters[sample, fixed_guide, variable_guide].append(quartet.R2)
            
    queue.put(('demux', quartet_name, chunk_number))
    
if __name__ == '__main__':
    base_dir = '/home/jah/projects/demux_test/data/2019_10_16_cut_sites'
    quartet_name = 'all'
    
    manager = multiprocessing.Manager()
    
    tasks_done_queue = manager.Queue()
    
    chunks_done = defaultdict(set) 
    
    chunkers_done = set()
    
    chunk_progress = tqdm.tqdm(desc='Chunk progress', total=10)
    demux_progress = tqdm.tqdm(desc='Demux progress', total=10)
    
    chunk_pool = multiprocessing.Pool(processes=4)
    demux_pool = multiprocessing.Pool(processes=10)
    
    with chunk_pool, demux_pool:
        chunk_results = []
        demux_results = []
        
        for which in fastq.quartet_order:
            args = (base_dir, quartet_name, which, tasks_done_queue)
            chunk_result = chunk_pool.apply_async(split_into_chunks, args)
            chunk_results.append(chunk_result)
            
        chunk_pool.close()
        
        while True:
            task_type, *task_info = tasks_done_queue.get()
            
            if task_type == 'chunk':
                c_quartet_name, c_which, c_chunk_id = task_info
                
                if c_chunk_id == 'DONE':
                    chunkers_done.add((c_quartet_name, c_which))
                else:
                    chunks_done[c_chunk_id].add((c_quartet_name, c_which))
                    if chunks_done[c_chunk_id] == {(c_quartet_name, which) for which in fastq.quartet_order}:
                        chunk_progress.update()
                        
                        args = (base_dir, c_quartet_name, c_chunk_id, tasks_done_queue)
                        demux_result = demux_pool.apply_async(demux_chunk, args)
                        demux_results.append(demux_result)

                if chunkers_done == {(quartet_name, which) for which in fastq.quartet_order}:
                    break
                    
            elif task_type == 'demux':
                demux_progress.update()
        
        while True:
            task_type, *task_info = tasks_done_queue.get()
            if task_type == 'demux':
                demux_progress.update()
                if demux_progress.n == chunk_progress.n:
                    break
            else:
                raise ValueError(task_type)
        
        chunk_pool.join()
        for chunk_result in chunk_results:
            if not chunk_result.successful():
                print(chunk_result.get())
                
        demux_pool.close()
                
        demux_pool.join()
        for demux_result in demux_results:
            if not demux_result.successful():
                print(demux_result.get())