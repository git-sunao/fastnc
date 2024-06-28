# fastnc
fast computation of natural components from given bispectrum.

Cite Sugiyama+2024 (in prep.)

## Installation
For now, you can install
```
python setup.py install
```
You can also install the pacakge from pip or conda:
```
pip install fastnc
```
or 
```
conda install ssunao::fastnc
```

## Get started
Tutorial notebook is available at [tutorial.ipynb](docs/tutorial.ipynb).

## Note on cache
This package create cache. The directory is at ~/.fastnc by default.
This can be changed by setting an environment variable `FASTNC_CACHE_DIR`. 
