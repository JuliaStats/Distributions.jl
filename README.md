Distributions.jl
================

[![Build Status](https://travis-ci.org/JuliaStats/Distributions.jl.png)](https://travis-ci.org/JuliaStats/Distributions.jl)

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

* Arcsine
* Bernoulli
* Beta
* Binomial
* Categorical
* Cauchy
* Chisq
* Dirichlet
* DiscreteUniform
* Exponential
* FDist
* Gamma
* Geometric
* HyperGeometric
* InvertedGamma
* InverseWishart
* Laplace
* Logistic
* logNormal
* MixtureModel
* Multinomial
* MultivariateNormal
* NegativeBinomial
* NoncentralBeta
* NoncentralChisq
* NoncentralF
* NoncentralT
* Normal
* Pareto
* Poisson
* Rayleigh
* TDist
* Uniform
* Weibull
* Wishart

## Simple Examples

    using Distributions

    x = rand(Normal(0.0, 1.0), 10_000)
    mean(x)

    d = Beta(1.0, 9.0)
    pdf(d, 0.9)
    quantile(d, 0.1)
    cdf(d, 0.1)

## Fit Distributions to Data using Maximum Likelihood Estimation

	using Distributions

    N = 100_000

    fit_mle(Bernoulli, rand(Bernoulli(0.7), N))

    fit_mle(Beta, rand(Beta(1.3, 3.7), N))

    fit_mle(Binomial, rand(Binomial(N, 0.3)), N)

    fit_mle(DiscreteUniform, rand(DiscreteUniform(300_000, 700_000), N))

    fit_mle(Exponential, rand(Exponential(0.1), N))

    fit_mle(Gamma, rand(Gamma(7.9, 3.1), N))

    fit_mle(Geometric, rand(Geometric(0.1), N))

    fit_mle(Laplace, rand(Laplace(10.0, 3.0), N))

    fit_mle(Normal, rand(Normal(11.3, 5.7), N))

    fit_mle(Poisson, rand(Poisson(19.0), N))

    fit_mle(Uniform, rand(Uniform(1.1, 98.3), N))
