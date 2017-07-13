# [Univariate Distributions](@id univariates)

*Univariate distributions* are the distributions whose variate forms are `Univariate` (*i.e* each sample is a scalar). Abstract types for univariate distributions:

```julia
const UnivariateDistribution{S<:ValueSupport} = Distribution{Univariate,S}

const DiscreteUnivariateDistribution   = Distribution{Univariate, Discrete}
const ContinuousUnivariateDistribution = Distribution{Univariate, Continuous}
```

## Common Interface

A series of methods are implemented for each univariate distribution, which provide useful functionalities such as moment computation, pdf evaluation, and sampling (*i.e.* random number generation).

### Parameter Retrieval

```@docs
params(::UnivariateDistribution)
succprob(::UnivariateDistribution)
failprob(::UnivariateDistribution)
scale(::UnivariateDistribution)
location(::UnivariateDistribution)
shape(::UnivariateDistribution)
rate(::UnivariateDistribution)
ncategories(::UnivariateDistribution)
ntrials(::UnivariateDistribution)
dof(::UnivariateDistribution)
```

**Note:** ``params`` are defined for all univariate distributions, while other parameter retrieval methods are only defined for those distributions for which these parameters make sense. See below for details.


### Computation of statistics

```@docs
maximum(::UnivariateDistribution)
minimum(::UnivariateDistribution)
mean(::UnivariateDistribution)
var(::UnivariateDistribution)
std(::UnivariateDistribution)
median(::UnivariateDistribution)
modes(::UnivariateDistribution)
mode(::UnivariateDistribution)
skewness(::UnivariateDistribution)
kurtosis(::UnivariateDistribution)
isplatykurtic(::UnivariateDistribution)
isleptokurtic(::UnivariateDistribution)
ismesokurtic(::UnivariateDistribution)
entropy(::UnivariateDistribution)
entropy(::UnivariateDistribution, ::Bool)
mgf(::UnivariateDistribution, ::Any)
cf(::UnivariateDistribution, ::Any)
```

### Probability Evaluation

```@docs
insupport(::UnivariateDistribution, x::Any)
pdf(::UnivariateDistribution, ::Real)
logpdf(::UnivariateDistribution, ::Real)
loglikelihood(::UnivariateDistribution, ::AbstractArray)
cdf(::UnivariateDistribution, ::Real)
logcdf(::UnivariateDistribution, ::Real)
ccdf(::UnivariateDistribution, ::Real)
logccdf(::UnivariateDistribution, ::Real)
quantile(::UnivariateDistribution, ::Real)
cquantile(::UnivariateDistribution, ::Real)
invlogcdf(::UnivariateDistribution, ::Real)
invlogccdf(::UnivariateDistribution, ::Real)
```

### Vectorized evaluation

Vectorized computation and inplace vectorized computation are supported for the following functions:

* [`pdf`](@ref)
* [`logpdf`](@ref)
* [`cdf`](@ref)
* [`logcdf`](@ref)
* [`ccdf`](@ref)
* [`logccdf`](@ref)
* [`quantile`](@ref)
* [`cquantile`](@ref)
* [`invlogcdf`](@ref)
* [`invlogccdf`](@ref)

For example, when `x` is an array, then `r = pdf(d, x)` returns an array `r` of the same size, such that `r[i] = pdf(d, x[i])`. One can also use `pdf!` to write results to pre-allocated storage, as `pdf!(r, d, x)`.


### Sampling (Random number generation)
```@docs
rand(::UnivariateDistribution)
rand!(::UnivariateDistribution, ::AbstractArray)
```

## Continuous Distributions

```@docs
Arcsine
Beta
BetaPrime
Biweight
Cauchy
Chi
Chisq
Cosine
Epanechnikov
Erlang
Exponential
FDist
Frechet
Gamma
GeneralizedExtremeValue
GeneralizedPareto
Gumbel
InverseGamma
InverseGaussian
Kolmogorov
KSDist
KSOneSided
Laplace
Levy
Logistic
LogNormal
NoncentralBeta
NoncentralChisq
NoncentralF
NoncentralT
Normal
NormalCanon
NormalInverseGaussian
Pareto
Rayleigh
SymTriangularDist
TDist
TriangularDist
Triweight
Uniform
VonMises
Weibull
```

## Discrete Distributions

```@docs
Bernoulli
BetaBinomial
Binomial
Categorical
DiscreteUniform
Geometric
Hypergeometric
NegativeBinomial
Poisson
PoissonBinomial
Skellam
```
