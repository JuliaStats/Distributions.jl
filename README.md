Distributions.jl
================

[![Build Status](https://travis-ci.org/JuliaStats/Distributions.jl.png)](https://travis-ci.org/JuliaStats/Distributions.jl)

A Julia package for probability distributions and associated functions. 

## Distribution Types

Each distribution is implemented by a Julia type, derived from an abstract type ``Distribution``. The distributions are classified into several categories, reified as a type hierarchy as follows:

```julia
abstract Distribution
abstract UnivariateDistribution             <: Distribution
abstract MultivariateDistribution           <: Distribution
abstract MatrixDistribution                 <: Distribution

abstract DiscreteUnivariateDistribution     <: UnivariateDistribution
abstract ContinuousUnivariateDistribution   <: UnivariateDistribution

abstract DiscreteMultivariateDistribution   <: MultivariateDistribution
abstract ContinuousMultivariateDistribution <: MultivariateDistribution

abstract ContinuousMatrixDistribution       <: MatrixDistribution
abstract DiscreteMatrixDistribution         <: MatrixDistribution

typealias NonMatrixDistribution Union(UnivariateDistribution, MultivariateDistribution)
typealias DiscreteDistribution Union(DiscreteUnivariateDistribution, DiscreteMultivariateDistribution)
typealias ContinuousDistribution Union(ContinuousUnivariateDistribution, ContinuousMultivariateDistribution)
```

#### Sample types

Generally, we observe following rules for sample types:

* Each sample from a univariate distribution is a scalar.
* Each sample from a multivariate distribution is a vector.
* Each sample from a matrix distribution is a matrix.
* The element type of a sample from a discrete distribution is ``Int``. 
* The element type of a sample from a continuous distribution is ``Float64``.
* Multiple samples of a univariate distribution are grouped into an array of arbitrary size.
* Samples of a multivariate distribution are grouped into a matrix where each column is a sample. For example, ``n`` samples of dimension ``d`` are grouped into a matrix of size ``(d, n)``.



#### Univariate Distributions

As of version 0.0.0, the package implements the following distributions:

**Discrete Distributions**:

* Bernoulli
* Binomial
* Categorical
* DiscreteUniform
* Geometric
* HyperGeometric
* NegativeBinomial
* Poisson
* Skellam

**Continuous Distributions**:

* Arcsine
* Beta
* BetaPrime
* Cauchy
* Chi
* Chisq
* Cosine
* Erlang
* Exponential
* FDist
* Gamma
* Gumbel
* InvertedGamma
* Laplace
* Levy
* Logistic
* LogNormal
* NoncentralBeta
* NoncentralChisq
* NoncentralF
* NoncentralT
* Normal
* Pareto
* Rayleigh
* TDist
* Uniform
* Weibull

#### Multivariate Distributions

* Dirichlet
* Multinomial
* MultivariateNormal

#### Matrix Distributions

* InverseWishart
* MixtureModel
* Wishart


## Functions

*Distributions.jl* also implements a variety of functions for evaluating statistics, computing pdf/logpdf, and sampling. 

#### Functions for univariate distributions

The methods below are typically implemented for univariate distributions. (**Note:** some functions may not be available for certain distributions.)

```julia
mean(d)           # expectation (mean) of the distribution d
var(d)            # variance
std(d)            # standard deviation
median(d)         # median value
skewness(d)       # skewness
kurtosis(d)       # excess kurtosis
entropy(d)        # entropy 
mgf(d, t)         # moment generating function
cf(d, t)          # characteristic function
modes(d)          # all modes (this function returns an array)

insupport(d, x)   # returns whether x is in the support of d
pdf(d, x)         # probability density/mass function
logpdf(d, x)      # logarithm of pdf

cdf(d, x)         # cumulative function
logcdf(d, x)      # logarithm of cdf
ccdf(d, x)        # complementary cumulative function, i.e. 1 - cdf(d, x)
logccdf(d, x)     # logarithm of ccdf

quantile(d, p)    # quantile, returns x such that cdf(x) == p
cquantile(d, p)   # complementary quantile, returns x such that ccdf(x) == p
invlogcdf(d, p)   # inverse logcdf
invlogccdf(d, p)  # inverse logccdf 
```

A number of functions, including ``pdf``, ``logpdf``, ``cdf``, ``logcdf``, ``ccdf``, ``logccdf``, ``invlogcdf``, and ``invlogccdf``, supports vectorized computation. Let ``f`` be one of these functions, and ``x`` be an array, one can write

```julia
r = f(d, x)        # returns an array of the same size of x, s.t. r[i] = f(d, x[i])
f!(r, d, x)        # writes the results to a pre-allocated array
```

Sampling from a distribution can be done using ``rand`` and ``rand!``:

```julia
rand(d)         # returns a value sampled from d
rand(d, n)      # returns a vector containing of n samples from d
rand(d, dims)   # returns a sample array of specific size

rand!(d, a)     # generates samples to a pre-allocated array a
```

#### Functions for Multivariate distributions

The following methods are implemented for multivariate distributions:

```julia
mean(d)         # mean vector
cov(d)          # covariance matrix
entropy(d)      # entropy

insupport(d, x)    # return whether x is in the support of d
pdf(d, x)          # probability density function
logpdf(d, x)       # logarithm of pdf

pdf!(r, d, x)      # evaluates pdf to a pre-allocated array.
logpdf!(r, d, x)   # evaluates logpdf to a pre-allocated array.
```

``pdf`` or ``logpdf`` returns a scalar when ``x`` is a vector, or returns a vector of length ``n`` when ``x`` is a matrix comprised of ``n`` columns. For ``pdf!`` and ``logpdf!``, ``r`` should have ``length(r) == size(x, 2)``. 

Like univariate distributions, sampling can be done using ``rand`` and ``rand!``:

```julia
rand(d)           # returns a sample vector
rand(d, n)        # returns a matrix of size (d, n): each column is a sample
rand!(d, x)       # generates sample(s) to a pre-allocated array x.
```


## Simple Examples

```julia
using Distributions

x = rand(Normal(0.0, 1.0), 10_000)
mean(x)

d = Beta(1.0, 9.0)
pdf(d, 0.9)
quantile(d, 0.1)
cdf(d, 0.1)
```

## Fit Distributions to Data using Maximum Likelihood Estimation

```julia
using Distributions

N = 100_000

fit_mle(Bernoulli, rand(Bernoulli(0.7), N))

fit_mle(Beta, rand(Beta(1.3, 3.7), N))

fit_mle(Binomial, 100, rand(Binomial(100, 0.3), N))

fit_mle(DiscreteUniform, rand(DiscreteUniform(300000, 700000), N))

fit_mle(Exponential, rand(Exponential(0.1), N))

fit_mle(Gamma, rand(Gamma(7.9, 3.1), N))

fit_mle(Geometric, rand(Geometric(0.1), N))

fit_mle(Laplace, rand(Laplace(10.0, 3.0), N))

fit_mle(Normal, rand(Normal(11.3, 5.7), N))

fit_mle(Poisson, rand(Poisson(19.0), N))

fit_mle(Uniform, rand(Uniform(1.1, 98.3), N))
```
