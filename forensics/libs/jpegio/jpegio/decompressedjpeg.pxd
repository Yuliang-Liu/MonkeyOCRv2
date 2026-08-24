cimport numpy as cnp
from jstruct cimport jstruct

cdef class DecompressedJpeg:

    cdef public list comp_info
    cdef public list quant_tables
    cdef public list coef_arrays
    cdef public list spatial_arrays
    cdef public list ac_huff_tables
    cdef public list dc_huff_tables
    cdef public list markers

    cdef jstruct* _jstruct_obj

    cdef _read_comp_info(self)
    cdef _read_markers(self)
    cdef _read_quant_tables(self)
    cdef _read_huffman_tables(self)
    cdef _read_dct_coefficients(self)

    cdef _write_markers(self)

    cpdef public read(self, fpath)
    cpdef public read_mem(self, mem)
    cpdef public write(self, fpath)
    cpdef get_coef_block(self, c, i, j)
    cpdef get_coef_block_array_shape(self, c)
    cpdef are_channel_sizes_same(self)
    cpdef count_nnz_ac(self)


    cdef _is_valid_fpath(self, fpath)

