import glob
import unittest
import os
from os.path import join as pjoin
from os.path import abspath as apath
import random

import numpy as np
import scipy.io as spio

import jpegio

BS = 8  # DCT block size


def create_list_fpaths(self):
    self.list_fpaths = []
    self.extensions = ["*.jpg", "*.jpeg"]

    dpath = os.path.dirname(__file__)
    for ext in self.extensions:
        for fpath in glob.glob(pjoin(dpath, "images", ext)):
            self.list_fpaths.append(apath(fpath))


def remove_modified_files(self):

    dpath = pjoin(os.path.dirname(__file__), "images")
    for entity in os.listdir(dpath):
        if "modified" in entity:
            fpath = pjoin(dpath, entity)
            os.remove(fpath)


class ComparisionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        create_list_fpaths(self)
        remove_modified_files(self)

    def test_repeat_read_100(self):
        """=> Check memory errors and garbage collection (100 iterations)."""
        for i in range(100):
            fpath = random.choice(self.list_fpaths)
            jpeg = jpegio.read(fpath)
            del jpeg

    @unittest.skip("This test takes a relatively long time.")
    def test_repeat_read_1000(self):
        """=> Check memory errors and garbage collection (1000 iterations)."""
        for i in range(1000):
            fpath = random.choice(self.list_fpaths)
            jpeg = jpegio.read(fpath)
            del jpeg

    def test_compare_dct_coef(self):
        """=> Test reading DCT coefficients."""
        for fpath in self.list_fpaths:
            fname = os.path.basename(fpath)
            dpath_mat = apath(
                pjoin(os.path.dirname(fpath), os.path.pardir, "matlab_outputs")
            )
            fpath_mat = pjoin(dpath_mat, "coef_arrays" + fname + ".mat")
            if not os.path.isfile(fpath_mat):
                continue

            mat = spio.loadmat(fpath_mat)
            coef_arrays_mat = mat["coef_arrays"][0]
            jpeg = jpegio.read(fpath)
            for i in range(len(jpeg.coef_arrays)):
                self.assertEqual(coef_arrays_mat[i].dtype, jpeg.coef_arrays[i].dtype)
                res = np.array_equal(jpeg.coef_arrays[i], coef_arrays_mat[i])
                self.assertTrue(res)

    def test_compare_coef_block_array_shape(self):
        """=> Test getting DCT block array shape."""
        for fpath in self.list_fpaths:
            jpeg = jpegio.read(fpath)

            for c in range(len(jpeg.coef_arrays)):
                coef_arr = jpeg.coef_arrays[c]
                blk_shape = jpeg.get_coef_block_array_shape(c)
                self.assertTrue(int(coef_arr.shape[0] / BS) == blk_shape[0])
                self.assertTrue(int(coef_arr.shape[1] / BS) == blk_shape[1])

    def test_compare_coef_block(self):
        """=> Test getting DCT block array."""
        for fpath in self.list_fpaths:
            jpeg = jpegio.read(fpath)

            for c in range(len(jpeg.coef_arrays)):
                coef_arr = jpeg.coef_arrays[c]
                nrows_blk, ncols_blk = jpeg.get_coef_block_array_shape(c)
                for i in range(nrows_blk):
                    for j in range(ncols_blk):
                        coef_blk = jpeg.get_coef_block(c, i, j)
                        self.assertTrue(
                            np.array_equal(
                                coef_arr[BS * i : BS * (i + 1), BS * j : BS * (j + 1)],
                                coef_blk,
                            )
                        )
                    # end of for
                # end of for
            # end of for
        # end of for

    def test_are_channel_sizes_same(self):
        """=> Test deciding sizes of all channels are identical."""
        dpath = os.path.dirname(__file__)

        # False cases
        jpeg = jpegio.read(pjoin(dpath, "images", "arborgreens01.jpg"))
        self.assertFalse(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "cherries01.jpg"))
        self.assertFalse(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "football01.jpg"))
        self.assertFalse(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "greenlake01.jpg"))
        self.assertFalse(jpeg.are_channel_sizes_same())

        # True cases
        jpeg = jpegio.read(pjoin(dpath, "images", "test01.jpg"))
        self.assertTrue(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "test02.jpg"))
        self.assertTrue(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "test03.jpg"))
        self.assertTrue(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "test04.jpg"))
        self.assertTrue(jpeg.are_channel_sizes_same())

        jpeg = jpegio.read(pjoin(dpath, "images", "test05.jpg"))
        self.assertTrue(jpeg.are_channel_sizes_same())

    def test_compare_count_nnz_ac(self):
        """=> Test counting non-zero DCT AC coefficients."""
        for fpath in self.list_fpaths:
            fname = os.path.basename(fpath)
            dpath_mat = apath(
                pjoin(os.path.dirname(fpath), os.path.pardir, "matlab_outputs")
            )
            fpath_mat = pjoin(dpath_mat, "nnz_" + fname + ".mat")
            if not os.path.isfile(fpath_mat):
                continue

            mat = spio.loadmat(fpath_mat)
            nnz_ac_mat = mat["nnz_ac"][0]

            jpeg = jpegio.read(fpath)
            nnz_ac_jpegio = jpeg.count_nnz_ac()

            self.assertTrue(nnz_ac_mat == nnz_ac_jpegio)
        # end of for

    # end of def


class WriteTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        create_list_fpaths(self)
        remove_modified_files(self)

    def test_write_dct_coef(self):
        """=> Test modifying a single DCT coefficient."""
        for fpath in self.list_fpaths:
            for i in range(3):  # Test 3 times
                jpeg = jpegio.read(fpath)
                fpath_no_ext, ext = os.path.splitext(fpath)
                fpath_modified = fpath_no_ext + "_modified" + ext

                ix_coef_arr = np.random.randint(0, len(jpeg.coef_arrays))
                coef_arr = jpeg.coef_arrays[ix_coef_arr]
                ix_row = np.random.randint(0, coef_arr.shape[0])
                ix_col = np.random.randint(0, coef_arr.shape[1])
                val = np.random.randint(-256, 256)

                coef_arr[ix_row, ix_col] = val

                self.assertTrue(hasattr(jpeg, "write"))
                jpeg.write(fpath_modified)
                jpeg_modified = jpegio.read(fpath_modified)

                coef_arr_modified = jpeg_modified.coef_arrays[ix_coef_arr]
                self.assertEqual(
                    coef_arr[ix_row, ix_col], coef_arr_modified[ix_row, ix_col]
                )

                del jpeg
                del jpeg_modified
                os.remove(fpath_modified)

    def test_write_quant_table(self):
        """=> Test modifying a single element of quantization tables."""
        for fpath in self.list_fpaths:
            for i in range(3):  # Test 3 times
                jpeg = jpegio.read(fpath)
                fpath_no_ext, ext = os.path.splitext(fpath)
                fpath_modified = fpath_no_ext + "_modified" + ext

                ix_qt = np.random.randint(0, len(jpeg.quant_tables))
                qt = jpeg.quant_tables[ix_qt]
                ix_row = np.random.randint(0, qt.shape[0])
                ix_col = np.random.randint(0, qt.shape[1])
                val = np.random.randint(1, 65535)

                qt[ix_row, ix_col] = val

                self.assertTrue(hasattr(jpeg, "write"))
                jpeg.write(fpath_modified)
                jpeg_modified = jpegio.read(fpath_modified)

                qt_modified = jpeg_modified.quant_tables[ix_qt]
                self.assertEqual(qt[ix_row, ix_col], qt_modified[ix_row, ix_col])

                del jpeg
                del jpeg_modified
                os.remove(fpath_modified)

    @unittest.skip("libjpeg cannot write arbitrarily modified Huffman table")
    def test_write_huffman_tables(self):
        """=> Test modifying a single element of Huffman tables."""
        for fpath in self.list_fpaths:
            for i in range(3):  # Test 3 times
                jpeg = jpegio.read(fpath)
                fpath_no_ext, ext = os.path.splitext(fpath)
                fpath_modified = fpath_no_ext + "_modified" + ext

                ix_hftb = np.random.randint(0, len(jpeg.ac_huff_tables))
                ac_hftb = jpeg.ac_huff_tables[ix_hftb]
                counts = ac_hftb["counts"]
                symbols = ac_hftb["symbols"]
                ix_counts = np.random.randint(0, counts.size)
                ix_symbols = np.random.randint(0, symbols.size)
                val_counts = np.random.randint(counts.min(), counts.max() + 1)
                val_symbols = np.random.randint(symbols.min(), symbols.max() + 1)

                print(counts)
                print(symbols)

                counts[ix_counts] = val_counts
                symbols[ix_symbols] = val_symbols

                print(counts)
                print(symbols)

                self.assertTrue(hasattr(jpeg, "write"))
                jpeg.write(fpath_modified)
                jpeg_modified = jpegio.read(fpath_modified)

                ac_hftb_modified = jpeg.ac_huff_tables[ix_hftb]
                counts_modified = ac_hftb_modified["counts"]
                symbols_modified = ac_hftb_modified["symbols"]
                self.assertEqual(counts[ix_counts], counts_modified[ix_counts])
                self.assertEqual(symbols[ix_symbols], symbols_modified[ix_symbols])

                del jpeg
                del jpeg_modified
                os.remove(fpath_modified)


if __name__ == "__main__":
    unittest.main()
