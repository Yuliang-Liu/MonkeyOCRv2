
from clibjpeg cimport JDIMENSION

cdef class ComponentInfo:    
    cdef public int component_id
    cdef public int h_samp_factor
    cdef public int v_samp_factor       
    cdef public int quant_tbl_no
    cdef public int ac_tbl_no
    cdef public int dc_tbl_no
    
    cdef public JDIMENSION downsampled_height         
    cdef public JDIMENSION downsampled_width
    
    cdef public JDIMENSION height_in_blocks
    cdef public JDIMENSION width_in_blocks