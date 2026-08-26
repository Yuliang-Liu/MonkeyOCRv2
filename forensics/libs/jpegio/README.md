# JpegIO

- A python package for accessing the internal variables of JPEG file format such as DCT coefficients and quantization tables.
- See also [jpeglib](https://github.com/martinbenes1996/jpeglib), which supports a comprehensive set of JPEG libraries.

## Installation

It is recommended to install by compiling yourself.
The installation process includes compiling C/C++ source codes.

```
python setup.py install
```

You can also use the pre-compiled wheels
(If some errors occur, try compiling the package using the above command).

On Linux:
```
pip install dist/jpegio-x.x.x-cp3x-cp3x-linux_<your architecture>.whl
```

On Windows 10 x64:
```
pip install dist/jpegio-x.x.x-cp3x-cp3x-win_amd64.whl
```

On macOS x64:
```
pip install dist/jpegio-x.x.x-cp3x-cp3x-macosx_<version>_x86_64.whl
```

## Making a wheel

You can use the following command for making a wheel for your own architecture such as x64 or ppc64le.

```
python setup.py bdist_wheel
```

The cooked wheel files are located at `dist` directory.


## Dependency
This package requires other packages.

- [`Cython`](https://cython.org/)
- [`NumPy`](http://www.numpy.org/)


## Usage example

```python
import jpegio as jio

jpeg = jio.read("image.jpg")
coef_array = jpeg.coef_arrays[0]  
quant_tbl = jpeg.quant_tables[0]

# Modifying jpeg.coef_arrays...
# Modifying jpeg.quant_tables...

jio.write(jpeg, "image_modified.jpg")
```

- `coef_arrays` is a list of `numpy.ndarray` objects that represent DCT coefficients of YCbCr channels in JPEG.
- `quant_tables` is a list of `numpy.ndarray` objects that represent the quantization tables in JPEG.

You can also utilize other variables (one of the simplest ways for finding them is to use `dir(img)`).
The names of member variables have been determined following the convention of libjpeg.

## References
- The core parts of this package, implemented in C/C++, are adopted from the souce codes of [Jessica Fridrich's laboratory](http://dde.binghamton.edu).
- The functionality of libjpeg is borrowed from [IJG](https://www.ijg.org/) and [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo).

## License
[Apache License 2.0](/LICENSE)



## Contributors
- [@dwgoon](https://github.com/dwgoon)
- [@detrin](https://github.com/detrin)
- [@martinbenes1996](https://github.com/martinbenes1996)
- [@EldritchJS](https://github.com/EldritchJS)
