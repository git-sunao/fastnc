# fastnc
fast computation of natural components from given bispectrum.

Cite Sugiyama+2024 [arxiv](http://arxiv.org/abs/2407.01798).

## Installation
From [pip](https://pypi.org/project/fastnc/)
```
pip install fastnc
```
From [conda](https://anaconda.org/ssunao/fastnc)
```
conda install ssunao::fastnc
```
You can also install the package manually after cloning this [fastnc](https://github.com/git-sunao/fastnc) repo
```
pip install .
```

## Getting started
Tutorial notebook is available at [tutorial.ipynb](docs/tutorial.ipynb).

## Note on cache
This package create cache. The directory is at ~/.fastnc by default.
This can be changed by setting an environment variable `FASTNC_CACHE_DIR`. 
