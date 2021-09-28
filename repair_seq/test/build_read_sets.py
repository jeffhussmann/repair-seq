import hits.sam

import repair_seq.pooled_screen
from repair_seq.test.test_read_sets import ReadSet, base_dir

prime_editing_pools = repair_seq.pooled_screen.get_all_pools('/lab/solexa_weissman/jah/projects/prime_editing_screens')

prime_editing_manual_read_sets = [
    ('PE2_screen', '2020_03_18_prime_editing_PE2_rep1', 'none', 'ABRAXAS1_1', [
        ('01:01101:001280:016266', ('wild type', 'clean'), 'wild type, no errors'),
        
        ('01:01101:001099:018897', ('wild type', 'mismatches'), 'wild type with 3 mismatches towards end'),
        ('01:01101:002600:012477', ('wild type', 'mismatches'), 'wild type with 1 mismatch towards middle'),
        
        ('01:01102:005819:022216', ('wild type', 'short indel far from cut'), 'wild type with 1 nt deletion towards middle'),
        ('01:01112:030020:002190', ('wild type', 'short indel far from cut'), 'wild type with 1 nt deletion towards end'),
        
        ('01:01101:001515:027383', ('intended edit', 'SNV'), 'intended SNV, no errors'),
        
        ('01:01102:026106:033082', ('intended edit', 'SNV + mismatches'), 'intended SNV plus 1 mismatch towards middle'),
        ('01:01103:011333:025222', ('intended edit', 'SNV + mismatches'), 'intended SNV plus 1 mismatch towards end'),
        
        ('01:01106:007346:030937', ('extension from intended annealing', 'n/a'), 'dominant extension from intended annealing product from screen'),
        
        ('01:01115:024261:032205', ('deletion', 'clean'), 'deletion, 15 nts near nick'),
        ('01:01117:014922:022200', ('deletion', 'clean'), 'deletion, 16 nts near nick'),
        
        ('01:01101:019153:036417', ('deletion', 'mismatches'), 'deletion, 93 nts spanning nick plus 1 mismatch towards end'),
        ('01:01153:022173:033724', ('deletion', 'mismatches'), 'deletion, 15 nts spanning nick plus 1 mismatch towards end'),
     ],
    ),
    ('PE3+50_screen', '2020_03_18_prime_editing_PE3+50_rep1', 'none', 'ABRAXAS1_1', [
        ('01:01101:001217:035916', ('wild type', 'clean'), 'wild type, no errors'),
        
        ('01:01101:023547:034507', ('unintended annealing of RT\'ed sequence', 'includes scaffold'), 'unintended rejoining, small amount of scaffold, rejoins with partially duplicated HA_RT'),
        ('01:01105:012093:004178', ('unintended annealing of RT\'ed sequence', 'includes scaffold'), 'unintended rejoining, medium amount of scaffold, rejoins with midway between nicks'),
        ('01:01105:012192:030968', ('unintended annealing of RT\'ed sequence', 'includes scaffold'), 'unintended rejoining, medium amount of scaffold, with 3 mismatches'),
        ('01:01105:012364:004022', ('unintended annealing of RT\'ed sequence', 'includes scaffold'), 'unintended rejoining, medium amount of scaffold, rejoins well past +50 nick'),
        ('01:01105:016206:031125', ('unintended annealing of RT\'ed sequence', 'includes scaffold'), 'unintended rejoining, medium amount of scaffold, with mismatches in scaffold and target'),
        
        ('01:01102:007401:007639', ('unintended annealing of RT\'ed sequence', 'includes scaffold, no SNV'), 'unintended rejoining, medium amount of scaffold but no intended SNV'),
        
        ('01:01107:012427:010551', ('unintended annealing of RT\'ed sequence', 'no scaffold'), 'unintended rejoining, no scaffold, rejoins with partially duplicated HA_RT'),
        ('01:01123:032551:017942', ('unintended annealing of RT\'ed sequence', 'no scaffold'), 'unintended rejoining, no scaffold, rejoins with fully duplicated HA_RT'),
        
        ('01:01101:003667:009283', ('duplication', 'simple'), 'duplication of entire 5\' overhang'),
        ('01:01101:023511:010739', ('duplication', 'simple'), 'duplication of part of 5\' overhang'),
        ('01:01102:006316:026428', ('duplication', 'simple'), 'duplication of part of 5\' overhang with a mismatch'),
        
        ('01:01101:017806:036714', ('edit + duplication', 'simple'), 'duplication with edit, no errors'),
        ('01:01103:013078:005603', ('edit + duplication', 'simple'), 'duplication with edit with 1 mismatch'),
        
        ('01:01105:032208:029716', ('edit + indel', 'edit + deletion'), 'consistent with deletion between nicks that preserves both nicking target, followed by intended edit'),
        
        ('01:01101:022779:005055', ('deletion', 'clean'), 'deletion, 17 nts between nicks'),
        
        ('01:01102:008558:017221', ('deletion', 'mismatches'), 'deletion, 17 nts between nicks plus 1 mismatch towards end'),
        
        ('01:01117:010411:027070', ('deletion + duplication', 'simple'), 'consistent with a duplication between nicks that preserves both nicking targets, followed by deletion'),
     ],
    ),
    ('PE3+50_screen', '2020_03_18_prime_editing_PE3-50_rep1', 'none', 'ABRAXAS1_1', [
        ('01:01101:001325:033786', ('duplication', 'simple'), 'duplication of part of 3\' overhang with mismatches'),
        ('01:01101:002682:016908', ('duplication', 'simple'), 'duplication of part of 3\' overhang with 1 mismatch toward beginning'),

        ('01:01101:004580:021668', ('duplication', 'iterated'), 'iterated duplication, no mismatches'),
        ('01:01101:007057:025394', ('duplication', 'iterated'), 'iterated duplication, no mismatches'),

        ('01:01101:006912:025394', ('duplication', 'complex'), 'complex duplication, no mismatches'),

        ('01:01101:004661:030483', ('deletion', 'clean'), 'deletion, 4 nts between nicks'),

        ('01:01101:001633:026960', ('deletion', 'mismatches'), 'deletion, 34 nts with multiple mismatches'),

        ('01:01101:009417:014481', ('deletion + duplication', 'simple'), 'deletion spanning -50 nick'),
        ('01:01104:026087:007216', ('deletion + duplication', 'iterated'), 'long deletion'),

        ('01:01101:023131:013119', ('edit + duplication', 'simple'), 'duplication of part of 3\' overhang with intended edit'),
     ],
    ),
]

def build_manual_read_sets():
    all_set_details = []

    for set_name, pool_name, fixed_guide, variable_guide, read_details in prime_editing_manual_read_sets:
        pool = prime_editing_pools[pool_name]
        
        read_set = ReadSet(set_name, pool.target_info.name)
        
        all_set_details.append({'set_name': read_set.set_name, 'target_info': read_set.target_info_name})
        
        exp = pool.single_guide_experiment(fixed_guide, variable_guide)
        header = hits.sam.get_header(exp.fns_by_read_type['bam_by_name']['collapsed_uncommon_R2'])
        
        alignment_sorter = hits.sam.AlignmentSorter(read_set.bam_fn, header, by_name=True)
        
        with open(read_set.expected_values_fn, 'w') as expected_values_fh, alignment_sorter:
            for read_id, (category, subcategory), description in read_details:
                als = exp.get_read_alignments(read_id)
                for al in als:
                    # Overwrite potential common sequence query_name. 
                    al.query_name = read_id
                    alignment_sorter.write(al)
                    
                expected_values_fh.write(f'{read_id}\t{category}\t{subcategory}\t{description}\n')
                
    with open(base_dir / 'read_set_details.txt', 'w') as fh:
        for set_details in all_set_details:
            fh.write(f'{set_details["set_name"]}\t{set_details["target_info"]}\n')