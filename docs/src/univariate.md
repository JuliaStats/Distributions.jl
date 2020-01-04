# [Univariate Distributions](@id univariates)

*Univariate distributions* are the distributions whose variate forms are `Univariate` (*i.e* each sample is a scalar). Abstract types for univariate distributions:

```julia
const UnivariateDistribution{S<:ValueSupport} = Distribution{Univariate,S}

const DiscreteUnivariateDistribution   = Distribution{Univariate, Discrete}
const ContinuousUnivariateDistribution = Distribution{Univariate, Continuous}
```

## Common Interface

A series of methods are implemented for each univariate distribution, which provide
useful functionalities such as moment computation, pdf evaluation, and sampling
(*i.e.* random number generation).

### Parameter Retrieval

**Note:** `params` are defined for all univariate distributions, while other parameter
retrieval methods are only defined for those distributions for which these parameters make sense.
See below for details.

```@docs
params(::UnivariateDistribution)
scale(::UnivariateDistribution)
location(::UnivariateDistribution)
shape(::UnivariateDistribution)
rate(::UnivariateDistribution)
ncategories(::UnivariateDistribution)
ntrials(::UnivariateDistribution)
dof(::UnivariateDistribution)
```

For distributions for which success and failure have a meaning,
the following methods are defined:
```@docs
succprob(::DiscreteUnivariateDistribution)
failprob(::DiscreteUnivariateDistribution)
```


### Computation of statistics

```@docs
maximum(::UnivariateDistribution)
minimum(::UnivariateDistribution)
extrema(::UnivariateDistribution)
mean(::UnivariateDistribution)
var(::UnivariateDistribution)
std(::UnivariateDistribution)
median(::UnivariateDistribution)
modes(::UnivariateDistribution)
mode(::UnivariateDistribution)
skewness(::UnivariateDistribution)
kurtosis(::UnivariateDistribution)
kurtosis(::Distribution, ::Bool)
isplatykurtic(::UnivariateDistribution)
isleptokurtic(::UnivariateDistribution)
ismesokurtic(::UnivariateDistribution)
entropy(::UnivariateDistribution)
entropy(::UnivariateDistribution, ::Bool)
entropy(::UnivariateDistribution, ::Real)
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
logdiffcdf(::UnivariateDistribution, ::T, ::T) where {T <: Real}
ccdf(::UnivariateDistribution, ::Real)
logccdf(::UnivariateDistribution, ::Real)
quantile(::UnivariateDistribution, ::Real)
cquantile(::UnivariateDistribution, ::Real)
invlogcdf(::UnivariateDistribution, ::Real)
invlogccdf(::UnivariateDistribution, ::Real)
```

### Sampling (Random number generation)
```@docs
rand(::AbstractRNG, ::UnivariateDistribution)
rand!(::AbstractRNG, ::UnivariateDistribution, ::AbstractArray)
```

## Continuous Distributions

```@docs
Arcsine
Beta
BetaPrime
Biweight
Cauchy
Chernoff
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
LocationScale
Logistic
LogitNormal
LogNormal
NoncentralBeta
NoncentralChisq
NoncentralF
NoncentralT
Normal
NormalCanon
NormalInverseGaussian
Pareto
PGeneralizedGaussian
Rayleigh
Semicircle
StudentizedRange
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
DiscreteNonParametric
Geometric
Hypergeometric
NegativeBinomial
Poisson
PoissonBinomial
Skellam
```

### Vectorized evaluation

Vectorized computation and inplace vectorized computation have been deprecated.
