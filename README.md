Distributions.jl
================

## Introduction

A Julia package for probability distributions and associated funtions. Each distribution is reified as a new type in the Julia type hierarchy deriving from the new abstract type, `Distribution`. These distributions minimally provide the following:

* `cdf`
* `pdf`
* `quantile`
* `rand`

Many distribution types also provide useful theoretical information about the distribution, such as:

* `mean`
* `median`
* `var`
* `std`

## Supported Distributions

As of v0.0.0, the following distributions have been implemented:

* Bernoulli
* Beta
* Binomial
* Categorical
* Cauchy
* Chisq
* Dirichlet
* Exponential
* FDist
* Gamma
* Geometric
* HyperGeometric
* Logistic
* logNormal
* Multinomial
* NegativeBinomial
* NoncentralBeta
* NoncentralChisq
* NoncentralF
* NoncentralT
* Normal
* Poisson
* TDist
* Uniform
* Weibull

## Simple Examples

    x = rand(Normal(0.0, 1.0), 10_000)
    mean(x)
    
    d = Beta(1.0, 9.0)
    pdf(d, 0.9)
    quantile(d, 0.1)
    cdf(d, 0.1)

