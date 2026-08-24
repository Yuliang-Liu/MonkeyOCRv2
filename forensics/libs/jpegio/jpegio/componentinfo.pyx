
cdef class ComponentInfo:
    
    def __cinit__(self):
        self.component_id = -1
        self.h_samp_factor = 0
        self.v_samp_factor = 0   
        self.quant_tbl_no = 0
        self.ac_tbl_no = 0
        self.dc_tbl_no = 0
        
        self.downsampled_height = 0
        self.downsampled_width = 0
        
        self.height_in_blocks = 0
        self.width_in_blocks = 0