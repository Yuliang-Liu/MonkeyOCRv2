import jpegio
from pathlib import Path


img_path = "./tests/images/arborgreens01.jpg"
assert Path(img_path).exists()

img_bytes = open(img_path, "rb").read()

jpg_read_by_bytes = jpegio.read(img_bytes)
print(jpg_read_by_bytes.quant_tables)
print(jpg_read_by_bytes.ac_huff_tables)
print(jpg_read_by_bytes.dc_huff_tables)
print(jpg_read_by_bytes.coef_arrays)


jpg_read_by_path = jpegio.read(img_path)

assert len(jpg_read_by_bytes.quant_tables) == len(jpg_read_by_path.quant_tables)
assert str(jpg_read_by_bytes.quant_tables) == str(jpg_read_by_path.quant_tables)
assert str(jpg_read_by_bytes.ac_huff_tables) == str(jpg_read_by_path.ac_huff_tables)
assert str(jpg_read_by_bytes.dc_huff_tables) == str(jpg_read_by_path.dc_huff_tables)
assert str(jpg_read_by_bytes.coef_arrays) == str(jpg_read_by_path.coef_arrays)