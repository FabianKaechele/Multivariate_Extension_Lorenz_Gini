# Multivariate_Extension_Lorenz_Gini
This repository contains the Python-Code for the paper: "A multivariate extension of the Lorenz curve based on copulas and a related multivariate Gini coefficient" (CITATION!!!).

# Features
* Calculate the MEGC
* Plot 2-dimensional MEILC
* Calculate X_star values of X
* Calculate univariate Gini coefficient

# Technologies
* Python version: 3.8
* Numpy version: 1.18.5
* Pandas version: 1.1.3
* Matplotlib version: 3.3.1
* Scipy verison: 1.5.0


# Installation
```
$pip install megcmeilc 
```
Noch zu machen!

# Usage

```
import megcmeilc 


# data is given

# calculate MEGC
mult_gini=megcmeilc.megc(data)

# plot MEILC (works only for 2 dimesional data)
megcmeilc.meilc(data)

# calculate X_star values
x_star=.egcmeilc.x_stern(data)

# calculate univariate Gini coefficient of onedimesional dataslice
gini_coef=megcmeilc.gini(datasclice)
```

