import numpy as np
import pandas as pd

def extract_numerator_and_denominator_counts(pool, numerator_outcomes, denominator_outcomes=None):
    if all(outcome in pool.category_counts.index for outcome in list(numerator_outcomes)):
        count_source = pool.category_counts
    elif all(outcome in pool.subcategory_counts.index for outcome in list(numerator_outcomes)):
        count_source = pool.subcategory_counts
    else:
        count_source = pool.outcome_counts('perfect')['none']
        
    numerator_counts = count_source.loc[numerator_outcomes].sum(axis='index')
        
    if denominator_outcomes is None:
        denominator_counts = pool.UMI_counts
    else:
        if not all(outcome in count_source.index for outcome in denominator_outcomes):
            raise ValueError(denominator_outcomes)
        denominator_counts = count_source.loc[denominator_outcomes].sum(axis='index')
        
    # Make sure neither counts has 'all_non_targeting' in index.
    for counts in [numerator_counts, denominator_counts]:
        counts.drop(['all_non_targeting', 'eGFP_NT2'], errors='ignore', inplace=True)
        
    return numerator_counts, denominator_counts

def compute_outcome_guide_statistics(pool, numerator_outcomes, denominator_outcomes=None):
    numerator_counts, denominator_counts = extract_numerator_and_denominator_counts(pool, numerator_outcomes, denominator_outcomes)

    frequencies = numerator_counts / denominator_counts
    
    nt_guides = pool.variable_guide_library.non_targeting_guides
    
    nt_numerator_counts = numerator_counts.loc[nt_guides]
    nt_denominator_counts = denominator_counts.loc[nt_guides]
    
    nt_fraction = nt_numerator_counts.sum() / nt_denominator_counts.sum()

    capped_fc = np.minimum(2**5, np.maximum(2**-5, frequencies / nt_fraction))

    guides_df = pd.DataFrame({
        'denominator_count': denominator_counts,
        'numerator_count': numerator_counts,
        'frequency': frequencies,
        'log2_fold_change': np.log2(capped_fc),
        'gene': pool.variable_guide_library.guides_df['gene'],
        'best_promoter': pool.variable_guide_library.guides_df['best_promoter'],
    })

    guides_df['non-targeting'] = guides_df.index.isin(pool.variable_guide_library.non_targeting_guides)
    
    guides_df.index.name = 'guide'

    for set_name, guides in pool.variable_guide_library.non_targeting_guide_sets.items():
        guides_df.loc[guides, 'gene'] = set_name

    guides_df.drop('eGFP_NT2', errors='ignore', inplace=True)
    
    return guides_df, nt_fraction

def convert_to_gene_statistics(guides_df, only_best_promoters=True): 
    gene_order = []
    
    for gene, rows in guides_df.groupby('gene'):
        gene_order.append(gene)

    all_means = {}

    for gene, rows in guides_df.query('best_promoter').groupby('gene')['log2_fold_change']:
        means = {}
        ordered = sorted(rows)
        ordered_by_abs = sorted(rows, key=np.abs, reverse=True)
        for n in [1, 2, 3]:
            means[f'lowest_{n}'] = np.mean(ordered[:n])
            means[f'highest_{n}'] = np.mean(ordered[-n:])
            means[f'extreme_{n}'] = np.mean(ordered_by_abs[:n])
            
        all_means[gene] = means
        
    genes_df = pd.DataFrame(all_means).T

    genes_df.index.name = 'gene'

    return genes_df
