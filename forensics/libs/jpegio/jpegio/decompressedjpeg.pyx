import os
import numpy as np
cimport numpy as cnp
from numpy import count_nonzero as cnt_nnz

import cython
from cython.operator cimport dereference as deref, preincrement as inc
from cython cimport view

from jpegio.clibjpeg cimport DCTSIZE, DCTSIZE2
from jpegio.jstruct cimport jstruct
from jpegio.jstruct cimport ptr_mat2D
from jpegio.jstruct cimport ptr_struct_ci
from jpegio.jstruct cimport ptr_struct_ht
from jpegio.componentinfo cimport ComponentInfo

from libc.stdio cimport printf

cdef class DecompressedJpeg:
    
    def __cinit__(self):
        self._jstruct_obj = NULL  # new jstruct()

    def __dealloc__(self):
        if self._jstruct_obj != NULL:
            del self._jstruct_obj
    
    cdef _is_valid_fpath(self, fpath):
        if not os.path.isfile(fpath):
            print("[JPEGIO] Wrong file path: %s" % (fpath))
            return False
        elif os.path.getsize(fpath) == 0:
            print("[JPEGIO] Empty file: %s" % (fpath))
            return False

        return True

    cpdef read(self, fpath):
        if not self._is_valid_fpath(fpath):
            return

        if self._jstruct_obj != NULL:
            del self._jstruct_obj

        self._jstruct_obj = new jstruct()
        self._jstruct_obj.jpeg_load(fpath.encode())
        self._jstruct_obj.spatial_load(fpath.encode())
        self._jstruct_obj.load_spatial = True

        self._read_comp_info()
        self._read_markers()
        self._read_quant_tables()
        self._read_huffman_tables()
        self._read_dct_coefficients()

    cpdef read_mem(self, mem):
        if self._jstruct_obj != NULL:
            del self._jstruct_obj

        self._jstruct_obj = new jstruct()
#         print("Begin")
        self._jstruct_obj.jpeg_load_mem(mem, len(mem))
#         print("Done")
#         self._jstruct_obj.spatial_load(fpath.encode())
#         self._jstruct_obj.load_spatial = True

        self._read_comp_info()
        self._read_markers()
        self._read_quant_tables()
        self._read_huffman_tables()
        self._read_dct_coefficients()


    cdef _read_comp_info(self):
        self.comp_info = list()

        cdef int i
        cdef int nch = self._jstruct_obj.num_components
        cdef ComponentInfo comp_info
        cdef ptr_struct_ci ptr_ci

        for i in range(nch):
            comp_info = ComponentInfo()
            ptr_ci = self._jstruct_obj.comp_info[i]

            comp_info.component_id = ptr_ci.component_id
            comp_info.h_samp_factor = ptr_ci.h_samp_factor
            comp_info.v_samp_factor = ptr_ci.v_samp_factor

            comp_info.quant_tbl_no = ptr_ci.quant_tbl_no
            comp_info.ac_tbl_no = ptr_ci.ac_tbl_no
            comp_info.dc_tbl_no = ptr_ci.dc_tbl_no

            comp_info.downsampled_height = ptr_ci.downsampled_height
            comp_info.downsampled_width = ptr_ci.downsampled_width

            comp_info.height_in_blocks = ptr_ci.height_in_blocks
            comp_info.width_in_blocks = ptr_ci.width_in_blocks

            self.comp_info.append(comp_info)
        # end of for

    cdef _read_markers(self):
        """Connect the buffer of markers to numpy.ndarray.
        """
        self.markers = list()
        cdef Py_ssize_t n_markers = self._jstruct_obj.markers.size()
        cdef Py_ssize_t i
        cdef bytes py_bytes
        cdef view.array cy_arr
        if n_markers > 0:
            for i in range(n_markers):
                py_bytes = self._jstruct_obj.markers[i]
                self.markers.append(py_bytes)

    cdef _read_quant_tables(self):
        """Connect the buffer of quantization tables to numpy.ndarray.
        """

        self.quant_tables = list()
        cdef int num_quant_tables = self._jstruct_obj.quant_tables.size()

        cdef ptr_mat2D ptr_mat2D_obj
        cdef view.array cy_arr
        cdef Py_ssize_t i
        for i in range(num_quant_tables):
            ptr_mat2D_obj = self._jstruct_obj.quant_tables[i]
            shape = (ptr_mat2D_obj.rows, ptr_mat2D_obj.cols)
            cy_arr = view.array(shape=shape,
                                itemsize=sizeof(int),
                                format="i",
                                mode="c",
                                allocate_buffer=False)
            cy_arr.data = <char *> ptr_mat2D_obj.GetBuffer()
            self.quant_tables.append(np.asarray(cy_arr))

    cdef _read_huffman_tables(self):
        """Connect the buffer of Huffman tables to numpy.ndarray.
        """

        self.ac_huff_tables = list()
        self.dc_huff_tables = list()

        cdef Py_ssize_t i
        cdef ptr_struct_ht ptr_ht
        cdef view.array cy_arr_counts
        cdef view.array cy_arr_symbols

        for i in range(self._jstruct_obj.ac_huff_tables.size()):
            ptr_ht = self._jstruct_obj.ac_huff_tables[i]
            cy_arr_counts = view.array(shape=(ptr_ht.counts.size(),),
                                       itemsize=sizeof(int),
                                       format="i",
                                       mode="c",
                                       allocate_buffer=False)
            cy_arr_counts.data = <char *> &(ptr_ht.counts[0])

            cy_arr_symbols = view.array(shape=(ptr_ht.symbols.size(),),
                                        itemsize=sizeof(int),
                                        format="i",
                                        mode="c",
                                        allocate_buffer=False)
            cy_arr_symbols.data = <char *> &(ptr_ht.symbols[0])
            self.ac_huff_tables.append({"counts": np.asarray(cy_arr_counts),
                                        "symbols": np.asarray(cy_arr_symbols)})

        for i in range(self._jstruct_obj.dc_huff_tables.size()):
            ptr_ht = self._jstruct_obj.dc_huff_tables[i]
            cy_arr_counts = view.array(shape=(ptr_ht.counts.size(),),
                                       itemsize=sizeof(int),
                                       format="i",
                                       mode="c",
                                       allocate_buffer=False)
            cy_arr_counts.data = <char *> &(ptr_ht.counts[0])

            cy_arr_symbols = view.array(shape=(ptr_ht.symbols.size(),),
                                        itemsize=sizeof(int),
                                        format="i",
                                        mode="c",
                                        allocate_buffer=False)
            cy_arr_symbols.data = <char *> &(ptr_ht.symbols[0])

            self.dc_huff_tables.append({"counts": np.asarray(cy_arr_counts),
                                        "symbols": np.asarray(cy_arr_symbols)})

    cdef _read_dct_coefficients(self):
        """Connect the buffer of DCT coefficients to numpy.ndarray.
        """
        self.coef_arrays = list()
        cdef ptr_mat2D ptr_mat2D_obj
        cdef view.array cy_arr
        cdef Py_ssize_t i
        for i in range(self._jstruct_obj.coef_arrays.size()):
            ptr_mat2D_obj = self._jstruct_obj.coef_arrays[i]
            shape = (ptr_mat2D_obj.rows, ptr_mat2D_obj.cols)
            cy_arr = view.array(shape=shape,
                                itemsize=sizeof(int),
                                format="i",
                                mode="c",
                                allocate_buffer=False)
            cy_arr.data = <char *> ptr_mat2D_obj.GetBuffer()
            self.coef_arrays.append(np.asarray(cy_arr))


    cpdef write(self, fpath):
        self._write_markers()
        self._jstruct_obj.jpeg_write(fpath.encode(), self.optimize_coding)

    cdef _write_markers(self):
        cdef Py_ssize_t n_markers = len(self.markers)
        cdef Py_ssize_t i
        cdef bytes py_bytes
        cdef char* cstr
        if n_markers > 0:

            self._jstruct_obj.markers.clear()
            for i in range(n_markers):
                cstr = <char *> self.markers[i]
                self._jstruct_obj.markers.push_back(cstr)
            # end of for
        # end of if

    cpdef get_coef_block(self, c, i, j):
        if not self.coef_arrays:
            raise AttributeError("coef_arrays has not been created yet.")

        cdef slice sr = slice(i * DCTSIZE, (i + 1) * DCTSIZE, 1)
        cdef slice sc = slice(j * DCTSIZE, (j + 1) * DCTSIZE, 1)
        return self.coef_arrays[c][sr, sc]

    cpdef get_coef_block_array_shape(self, c):
        if not self.coef_arrays:
            raise AttributeError("coef_arrays has not been created yet.")

        return (int(self.coef_arrays[c].shape[0] / DCTSIZE),
                int(self.coef_arrays[c].shape[1] / DCTSIZE))

    cpdef are_channel_sizes_same(self):
        cdef ComponentInfo ci
        cdef set set_nrows = set()
        cdef set set_ncols = set()

        for ci in self.comp_info:
            if len(set_nrows) == 1 and ci.downsampled_height not in set_nrows:
                return False
            set_nrows.add(ci.downsampled_height)

            if len(set_ncols) == 1 and ci.downsampled_width not in set_ncols:
                return False
            set_ncols.add(ci.downsampled_width)

        return True

    cpdef count_nnz_ac(self):
        num_nnz_ac = 0
        for i in range(self.num_components):
            coef = self.coef_arrays[i]
            num_nnz_ac += (cnt_nnz(coef) - cnt_nnz(coef[0::8, 0::8]))
        return num_nnz_ac

    @property
    def image_width(self):
        return self._jstruct_obj.image_width

    @property
    def image_height(self):
        return self._jstruct_obj.image_height

    @property
    def image_components(self):
        return self._jstruct_obj.image_components

    @property
    def image_color_space(self):
        return self._jstruct_obj.image_color_space

    @property
    def num_components(self):
        return self._jstruct_obj.num_components

    @property
    def jpeg_color_space(self):
        return self._jstruct_obj.jpeg_color_space

    @property
    def optimize_coding(self):
        return self._jstruct_obj.optimize_coding

    @property
    def progressive_mode(self):
        return self._jstruct_obj.progressive_mode
