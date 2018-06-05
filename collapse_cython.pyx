cdef int OFFSET = 33

def hamming_distance(char* first, char* second):
    cdef int i
    cdef int d = 0
    cdef int length = len(first)
    
    for i in range(length):
        if first[i] != second[i]:
            d += 1
            
    return d

def hq_hamming_distance(char* first_seq, char* second_seq, char* first_qual, char* second_qual, int min_q):
    cdef int i
    cdef int d = 0
    cdef int length = len(first_seq)
    cdef int floor = min_q + OFFSET
    
    for i in range(length):
        if (first_seq[i] != second_seq[i]) and (first_qual[i] >= floor) and (second_qual[i] >= floor):
            d += 1
            
    return d

def hq_mismatches_from_seed(char* seed, char* seq, char[:] qual, int min_q):
    cdef int i
    cdef int d = 0
    cdef int length = len(seq)
    cdef int floor = min_q
    
    for i in range(length):
        if (seq[i] != seed[i]) and (qual[i] >= floor):
            d += 1
            
    return d
