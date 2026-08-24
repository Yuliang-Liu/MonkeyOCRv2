import unittest
import glob
import os
from os.path import join as pjoin
from os.path import abspath as apath
import random

import jpegio


def create_list_fpaths(self):
    self.list_fpaths = []
    self.extensions = ["*.jpg", "*.jpeg"]
    self.extensions.extend([ext.upper() for ext in self.extensions])

    dpath = apath(os.path.dirname(__file__))
    for ext in self.extensions:
        for fpath in glob.glob(pjoin(dpath, "stegoappdb", "covers", ext)):
            self.list_fpaths.append(apath(fpath))

        for fpath in glob.glob(pjoin(dpath, "stegoappdb", "stegos", ext)):
            self.list_fpaths.append(apath(fpath))


class ReadTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        create_list_fpaths(self)

    def test_read_stegoappdb_jpeg(self):
        """=> Test reading StegoAppDB JPEG files (1000 iterations)."""
        for i in range(1000):
            fpath = random.choice(self.list_fpaths)
            jpeg = jpegio.read(fpath)


if __name__ == "__main__":
    unittest.main()
