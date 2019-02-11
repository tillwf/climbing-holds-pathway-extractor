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
python extractor images/img.jpg
```

It will create a file `images/img_<color>.jpg` for each color detected (from the 6 decribed earlier).


## Tweak


#### Colors

You can add or remove color by editing the color range of the global variable ``. It uses the HSV representation (hue, saturation, lightness).

#### Blur

A bluring effect is added to ease the readability, for the black holds in particular, but you can easily remove it.


