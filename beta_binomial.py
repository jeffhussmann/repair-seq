from collections import defaultdict

import numpy as np
import mpmath
import pandas as pd
import scipy.stats
import scipy.optimize
import scipy.special

n_choose_k = scipy.special.comb

def fit_beta_binomial(numerator_counts, denominator_counts):
    p = numerator_counts.sum() / denominator_counts.sum()
    
    def negative_log_likelihood(a):
        b = a / p - a
        count_pairs = zip(numerator_counts, denominator_counts)
        return -sum(scipy.stats.betabinom.logpmf(n, d, a, b) for n, d in count_pairs)
    
    results = scipy.optimize.minimize_scalar(negative_log_likelihood, bounds=(0, None))
    
    if not results.success:
        raise ValueError(results)
    else:
        alpha = results.x
        beta = alpha / p - alpha
        return alpha, beta

def beta_binomial_pvals(numerator, denominator, alpha, beta):
    # Note: sf required to maintain precision.
    # Since sf is P(> n), outcome_count - 1 required to produce P(>= n)
    pvals = {
        'down': scipy.stats.betabinom.cdf(numerator, denominator, alpha, beta),
        'up': scipy.stats.betabinom.sf(numerator - 1, denominator, alpha, beta),
    }
    return pvals

def p_k_of_n_less(n, k, sorted_ps):
    if k > n:
        return 1
    else:
        a = sorted_ps[k - 1]
        
        total = 0
        for free_dimensions in range(0, n - k + 1):
            total += n_choose_k(n, free_dimensions) * (1 - a)**free_dimensions * a**(n - free_dimensions)
        
        return total

def compute_outcome_guide_statistics(pool, numerator_outcomes, denominator_outcomes=None):
    numerator_counts, denominator_counts = extract_numerator_and_denominator_counts(pool, numerator_outcomes, denominator_outcomes)

    frequencies = numerator_counts / denominator_counts
    
    nt_guides = pool.variable_guide_library.non_targeting_guides
    
    nt_numerator_counts = numerator_counts.loc[nt_guides]
    nt_denominator_counts = denominator_counts.loc[nt_guides]
    
    nt_fraction = nt_numerator_counts.sum() / nt_denominator_counts.sum()

    alpha, beta = fit_beta_binomial(nt_numerator_counts, nt_denominator_counts)
    
    ps = beta_binomial_pvals(numerator_counts, denominator_counts, alpha, beta)

    genes = pool.variable_guide_library.guides_df['gene']

    capped_fc = np.minimum(2**5, np.maximum(2**-5, frequencies / nt_fraction))

    guides_df = pd.DataFrame({'denominator_count': denominator_counts,
                              'numerator_count': numerator_counts,
                              'frequency': frequencies,
                              'log2_fold_change': np.log2(capped_fc),
                              'p_down': ps['down'],
                              'p_up': ps['up'],
                              'gene': genes,
                             })
    
    guides_df.index.name = 'guide'
    
    recalculate_low_precision_pvals(guides_df, alpha, beta)
    
    # Bonferonni factor of 2
    guides_df['p_relevant'] = guides_df[['p_down', 'p_up']].min(axis=1) * 2
    guides_df['-log10_p_relevant'] = -np.log10(np.maximum(np.finfo(np.float64).tiny, guides_df['p_relevant']))

    return guides_df, nt_fraction

def recalculate_low_precision_pvals(guides_df, alpha, beta):
    for guide, row in guides_df.iterrows():
        if row['p_down'] < 1e-8:
            B = scipy.stats.betabinom(row['denominator_count'], alpha, beta)
            p_down = float(mpmath.fsum([mpmath.exp(B.logpmf(v)) for v in range(row['numerator_count'] + 1)]))
            guides_df.loc[guide, 'p_down'] = p_down
            
        if row['p_up'] < 1e-8:
            B = scipy.stats.betabinom(row['denominator_count'], alpha, beta)
            p_up = float(mpmath.fsum([mpmath.exp(B.logpmf(v)) for v in range(row['numerator_count'], row['denominator_count'])]))
            guides_df.loc[guide, 'p_up'] = p_up

def compute_outcome_gene_statistics(guides_df): 
    ps = defaultdict(list)

    max_k = 9

    gene_order = []
    
    for gene, rows in guides_df.groupby('gene'):
        if gene == 'negative_control':
            continue
            
        gene_order.append(gene)
        
        for direction in ('down', 'up'):
            sorted_ps = sorted(rows[f'p_{direction}'].values)
            n = len(sorted_ps)
            for k in range(1, max_k + 1):
                ps[direction, k].append(p_k_of_n_less(n, k, sorted_ps))
            
    uncorrected_ps_df = pd.DataFrame(ps, index=gene_order).min(axis=1, level=0)
    
    guides_per_gene = guides_df.groupby('gene').size()
    bonferonni_factor = np.minimum(max_k, guides_per_gene)
    corrected_ps_df = np.minimum(1, uncorrected_ps_df.multiply(bonferonni_factor, axis=0))

    up_genes = corrected_ps_df.query('up < down')['up'].sort_values(ascending=False).index

    grouped_fcs = guides_df.groupby('gene')['log2_fold_change']

    gene_log2_fold_changes = pd.DataFrame({
        'up': grouped_fcs.nlargest(2).mean(level=0),
        'down': grouped_fcs.nsmallest(2).mean(level=0),
    })

    gene_log2_fold_changes['relevant'] = gene_log2_fold_changes['down']
    gene_log2_fold_changes.loc[up_genes, 'relevant'] = gene_log2_fold_changes.loc[up_genes, 'up']

    corrected_ps_df['relevant'] = corrected_ps_df['down']
    corrected_ps_df.loc[up_genes, 'relevant'] = corrected_ps_df.loc[up_genes, 'up']

    genes_df = pd.concat({'log2_fold_change': gene_log2_fold_changes, 'p': corrected_ps_df}, axis=1)

    negative_log10_p = -np.log10(np.maximum(np.finfo(np.float64).tiny, genes_df['p']))
    negative_log10_p = pd.concat({'-log10_p': negative_log10_p}, axis=1)

    genes_df = pd.concat([genes_df, negative_log10_p], axis=1)

    genes_df.index.name = 'gene'

    return genes_df