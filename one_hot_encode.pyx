# Made by Michael Wainberg

# Turn on all optimization options
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: cdivision=True

# Enable warnings
# cython: warn.undeclared=True
# cython: warn.maybe_uninitialized=True
# cython: warn.unused=True
# cython: warn.unused_arg=True
# cython: warn.unused_result=True

# A lookup table is marginally faster than if statements
# To generate:
# mapping = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3}
# print [mapping.get(chr(character_code), -1) for character_code in range(256)]

cdef int *encode_base = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 1, -1, -1,
                         -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,
                         -1, 1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1]

# cdef int encode_base(unsigned char base) nogil:
#     if base == 'A' or base == 'a':
#         return 0
#     if base == 'C' or base == 'c':
#         return 1
#     if base == 'G' or base == 'g':
#         return 2
#     if base == 'T' or base == 't':
#         return 3
#     else:
#         return -1

def one_hot_encode(char *sequence, char[:, :] encoded_sequence):
    cdef int encoding_index, sequence_index, sequence_length = len(sequence)
    with nogil:
        for sequence_index in range(sequence_length):
            encoding_index = encode_base[<int> sequence[sequence_index]]
            if encoding_index != -1:  # treat invalid letters as absent bases
                encoded_sequence[encoding_index, sequence_index] = 1

