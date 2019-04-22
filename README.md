Distributions.jl
================

[![Travis status](https://travis-ci.org/JuliaStats/Distributions.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distributions.jl)
[![Appveyor status](https://ci.appveyor.com/api/projects/status/xqm07gyruflhnha7/branch/master?svg=true)](https://ci.appveyor.com/project/simonbyrne/distributions-jl/branch/master)
[![](https://zenodo.org/badge/DOI/10.5281/zenodo.2647520.svg)](https://zenodo.org/record/2647520)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/Distributions.jl/badge.svg?branch=master)](https://coveralls.io/r/JuliaStats/Distributions.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaStats.github.io/Distributions.jl/latest/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaStats.github.io/Distributions.jl/stable/)

A Julia package for probability distributions and associated functions. Particularly, Distributions implements:

* Moments (e.g mean, variance, skewness, and kurtosis), entropy, and other properties
* Probability density/mass functions (pdf) and their logarithm (logpdf)
* Moment generating functions and characteristic functions
* Sampling from population or from a distribution
* Maximum likelihood estimation

**Note:** The functionalities related to conjugate priors have been moved to the [ConjugatePriors package](https://github.com/JuliaStats/ConjugatePriors.jl).


#### Resources

* **Documentation**: <https://JuliaStats.github.io/Distributions.jl/stable/>

## Citing

Use the most recent DOI badge above, cite as:
```
@misc{dahua_lin_2019_2647520,
  author       = {Dahua Lin and
                  John Myles White and
                  Simon Byrne and
                  Douglas Bates and
                  Andreas Noack and
                  John Pearson and
                  Alex Arslan and
                  Kevin Squire and
                  David Anthoff and
                  Theodore Papamarkou and
                  Mathieu Besançon and
                  Jan Drugowitsch and
                  Moritz Schauer and
                  Avik Sengupta and
                  Brian J Smith and
                  Giuseppe Ragusa and
                  Alexey Stukalov and
                  Gord Stephen and
                  Christoph Dann and
                  micklat and
                  Martin O'Leary and
                  Mike J Innes and
                  Jiahao Chen and
                  Iain Dunning and
                  Gustavo Lacerda and
                  Yichao Yu and
                  Simon Kornblith and
                  Richard Reeve and
                  Rory Finnegan and
                  Kai Xu},
  title        = {{JuliaStats/Distributions.jl: Registered version 
                   with Project file}},
  month        = apr,
  year         = 2019,
  doi          = {10.5281/zenodo.2647520},
  url          = {https://doi.org/10.5281/zenodo.2647520}
}
```
