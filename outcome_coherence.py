#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

import yaml
import numpy as np
import pandas as pd

from knockin import experiment
from sequencing import fastq
from sequencing.utilities import group_by
import collapse

class UMI_Outcome(object):
    def __init__(self, name, guide, category, subcategory, details):
        annotation = collapse.cluster_Annotation.from_identifier(name)
        for k, v in annotation.items():
            setattr(self, k, v)
        
        self.guide = guide
        self.category = category
        self.subcategory = subcategory
        self.details = details
        self.name = name
    
    @property
    def outcome(self):
        return (self.category, self.subcategory, self.details)        

class cell_Outcome(object):
    def __init__(self, UMI_outcome, UMIs, coherence):
        self.guide = UMI_outcome.guide
        self.cell_BC = UMI_outcome.cell_BC
        self.num_UMIs = len(UMIs)
        self.num_reads = sum(u.num_reads for u in UMIs)
        self.category = UMI_outcome.category
        self.subcategory = UMI_outcome.subcategory
        self.details = UMI_outcome.details
        self.coherence = coherence
        self.UMIs = [u.UMI for u in UMIs]
        self.multiplet = is_multiplet[self.cell_BC]
    
    @property
    def outcome(self):
        return (self.category, self.subcategory, self.details)        

def load_UMIs(exp):
    UMIs = []
    
    guide = exp.name
    
    for line in exp.fns['outcome_list'].open():
        name, category, subcategory, details = line.strip().split('\t')
        
        UMI = UMI_Outcome(name, guide, category, subcategory, details)
        UMIs.append(UMI)
        
    UMIs = sorted(UMIs, key=lambda u: (u.guide, u.cell_BC, u.UMI))
        
    return UMIs

def make_UMIs_table(UMIs):
    columns = [
        'guide',
        'cell_BC',
        'UMI',
        'num_reads',
        'category',
        'subcategory',
        'details',
        'name',
    ]
    rows = []
    for UMI in UMIs:
        row = [getattr(UMI, k) for k in columns]
        rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    return table

def make_cells_table(cells):
    columns = [
        'guide',
        'cell_BC',
        'multiplet',
        'coherence',
        'num_UMIs',
        'num_reads',
        'category',
        'subcategory',
        'details',
    ]
    
    rows = []
    for cell in cells:
        row = [getattr(cell, k) for k in columns]
        rows.append(row)
    table = pd.DataFrame(rows, columns=columns)
    return table

def UMI_coherence(all_UMIs):
    UMIs = {
        'coherent': [],
        'incoherent': [],
    }

    for cell_BC, cell_UMIs in group_by(all_UMIs, lambda u: u.cell_BC):
        for outcome, outcome_UMIs in group_by(cell_UMIs, lambda u: u.outcome, sort=True):
            collapse.error_correct_UMIs(outcome_UMIs)

        for UMI, UMI_outcomes in group_by(cell_UMIs, lambda u: u.UMI, sort=True):
            all_outcomes = set(u.outcome for u in UMI_outcomes)

            if len(all_outcomes) == 1:
                relevant_outcomes = all_outcomes
                coherence = 'coherent'

            else:
                for category, subcategory, details in list(all_outcomes):
                    if category == 'bad sequence':
                        all_outcomes.remove((category, subcategory, details))
                    elif (category, subcategory, details) == ('no indel', 'other', 'ambiguous'):
                        all_outcomes.remove((category, subcategory, details))

                relevant_outcomes = all_outcomes

                if len(relevant_outcomes) == 1:
                    coherence = 'coherent'
                else:
                    coherence = 'incoherent'

            for outcome in relevant_outcomes:
                relevant = [u for u in UMI_outcomes if u.outcome == outcome]
                UMI_outcome = relevant[0]
                UMI_outcome.num_reads = sum(u.num_reads for u in relevant)

                UMIs[coherence].append(UMI_outcome)
                
    return UMIs

def cell_coherence(UMIs):
    cell_outcomes = []
    
    for cell_BC, cell_UMIs in group_by(UMIs, lambda u: u.cell_BC):
        all_outcomes = set(u.outcome for u in cell_UMIs)
 
        for category, subcategory, details in list(all_outcomes):
            if category in ['bad sequence']:
                all_outcomes.remove((category, subcategory, details))
        
        interesting_outcomes = set((category, subcategory, details) for category, subcategory, details in all_outcomes if (category != 'endogenous' and details != 'ambiguous'))
        
        if len(interesting_outcomes) == 0:
            coherence = 'no capture'
        
        elif len(interesting_outcomes) == 1:
            coherence = 'coherent'

        else:
            coherence = 'incoherent'

        for outcome in all_outcomes:
            relevant = [u for u in cell_UMIs if u.outcome == outcome]
            num_UMIs = len(relevant)
            num_reads = sum(u.num_reads for u in relevant)
            cell_outcome = cell_Outcome(relevant[0], relevant, coherence)
            cell_outcomes.append(cell_outcome)
            
    return cell_outcomes

is_multiplet = pd.read_csv('/home/jah/projects/britt/data/cell_identities.csv').set_index('cell_barcode')['number_of_cells'] > 1

def process(exp):
    all_UMIs = load_UMIs(exp)
    UMIs = UMI_coherence(all_UMIs)
    cell_outcomes = cell_coherence(UMIs['coherent'])

    UMI_tables = {k: make_UMIs_table(UMIs[k]) for k in UMIs}

    for k in UMI_tables:
        fn = exp.dir / 'UMIs_{}.txt'.format(k)
        UMI_tables[k].to_csv(fn, index=False, sep='\t')

    cell_table = make_cells_table(cell_outcomes)

    fn = exp.dir / 'cells.txt'
    cell_table.to_csv(fn, index=False, sep='\t')


def load_cell_tables(base_dir, group):
    exps = experiment.get_all_experiments(base_dir, {'group': group})
    fns = [exp.dir / 'cells.txt' for exp in exps]
    tables = [pd.read_table(fn) for fn in sorted(fns)]
    cell_table = pd.concat(tables, ignore_index=True)
    return cell_table

def load_UMI_table(base_dir, group, name):
    exp = experiment.Experiment(base_dir, group, name)
    fn = exp.dir / 'UMIs_coherent.txt'
    table = pd.read_table(fn)
    return table

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument('--process', nargs=2, metavar=('GROUP', 'NAME'))
    mode_group.add_argument('--parallel', metavar='MAX_PROCS')

    parser.add_argument('--conditions')

    args = parser.parse_args()

    if args.conditions is None:
        conditions = {}
    else:
        conditions = yaml.load(args.conditions)

    base_dir = Path('/home/jah/projects/britt')

    exps = experiment.get_all_experiments(base_dir, conditions)

    arg_tuples = sorted((exp.group, exp.name) for exp in exps)

    if args.parallel is not None:
        parallel_command = [
            'parallel',
            '-n', '2', 
            '--verbose',
            '--max-procs', args.parallel,
            './outcome_coherence.py',
            '--process', ':::',
        ]

        for arg_tuple in arg_tuples:
            parallel_command.extend(arg_tuple)
    
        subprocess.check_call(parallel_command)

    elif args.process is not None:
        group, name = args.process
        exp = experiment.Experiment(base_dir, group, name)
        process(exp)

