# Multivariate_Extension_Lorenz_Gini
This repository contains the Python-Code for the paper: 

***A multivariate extension of the Lorenz curve based on copulas and a related multivariate Gini coefficient.***

 published in the Journal of Economic Inequality (2022).
[Link (klick!) to the paper.](https://doi.org/10.1007/s10888-022-09533-x)

## Features
* Calculate the MEGC
* Plot 2-dimensional MEILC
* Calculate X_star values of X
* Calculate univariate Gini coefficient

## Technologies
* Python version: 3.8
* Numpy version: 1.18.5
* Pandas version: 1.1.3
* Matplotlib version: 3.3.1
* Scipy verison: 1.5.0

## Usage
save MEILC_MEGC.py in project

*data*:  np-array  (nxd) of given data

```python

import MEILC_MEGC as megcmeilc

# calculate MEGC
mult_gini = megcmeilc.megc(data)

# plot MEILC (works only for 2 dimensional data)
megcmeilc.meilc(data)

# calculate X_star values
X_star = megcmeilc.x_star(data)

# calculate univariate Gini coefficient of onedimensional dataslice
gini_coef = megcmeilc.gini(datasclice)
```

