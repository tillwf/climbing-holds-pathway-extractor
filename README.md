# Climbing holds pathway extractor

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Extract pathway from climbing wall picture based on holds color.
This program was first design to work on Arkose picture from the site https://www.sboulder.com/ and thus by default detect only the colors: yellow, green, blue, red, black and purple.

Inspired from https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

**Example:**:

![Animation example](https://github.com/tillwf/climbing-holds-pathway-extractor/blob/master/examples/n7mHJwtRJxf52L9dw.gif)

## Installation


```
pip install -r requirements.txt
```


## Usage

```
python extractor.py -i examples/iQnnQ47T9ZnpxWMJf.jpg -e hed
```

It will create a file `examples/iQnnQ47T9ZnpxWMJf_<color>.jpg` for each color detected (from the 6 decribed earlier).

There is another parameter `-e` or `--edge-detector` which can be `canny` or `hed` (see Sources) to specify the edge detector algorithm in the background removal process. If no edge detector is specified, the program will only apply the color filter, and not the background removal part. 

### Brackground removal

If the edge detector chosen is `hed`, this file must be download and put inside the folder `hed_model`: [hed_pretrained_bsds.caffemodel](http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel)

## Tweak


#### Colors

You can add or remove color by editing the color range of the global variable ``. It uses the HSV representation (hue, saturation, lightness).

Notes: For the `red` color, two ranges are used. The ones below and above 0. 

#### Blur

A bluring effect is added to ease the readability, for the black holds in particular, but you can easily remove it.


## Ideas

 - Cut as much as possible the holds. Then cluster the colors using `BIRCH` algorithm and filter on the closest color in input
 - Take the full picture and apply `DBSCAN` algorithm to partition the picture and filter on the closest color in input

## Sources

 - [Holistically-Nested Edge Detection](https://github.com/s9xie/hed)
 - [Zero-parameter, automatic Canny edge detection with Python and OpenCV](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
 - [Holistically-Nested Edge Detection with OpenCV and Deep Learning](https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/)
