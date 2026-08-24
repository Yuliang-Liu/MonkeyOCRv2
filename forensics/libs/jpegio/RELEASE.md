# Release 0.2.2

## Major Features and Improvements
* The type of `DecompressedJpeg.markers` is changed to `List[bytes]` from `List[str]`.
  From now on, decoding the markers is up to users.

# Release 0.2.0

## Major Features and Improvements
* ``DecompressedJpeg`` supports both reading and writing some internal variables of JPEG file format.
* The core of `DecompressedJpeg` has been changed to `jstruct` and `mat2D` in C++, which are adopted from [Fridrich's lab.](http://dde.binghamton.edu), and partly modified for Python wrapping.
* `ZigZagDct1d` is removed and will be updated again in the future.
* A test case for reading StegoAppDB JPEG files has added.
* The type of `DecompressedJpeg.quant_tables` is changed to list from `numpy.ndarray`.

# Release 0.1.3

## Major Features and Improvements
* `DecompressedJpeg` supports only reading JPEG files, not writing.
* `DecompressedJpeg` depends on the functionality of libjpeg (linux and macos) or libjpeg-turbo (windows).
* `ZigZagDct1d` efficiently reads DCT coefficients in the zig-zag way and presents it as `numpy.ndarray`.
