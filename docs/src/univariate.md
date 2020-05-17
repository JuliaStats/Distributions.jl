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
```
Density plot `Arcsine(0, 1)`

```@example Arcsine
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Arcsine(0,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(ymin = 0, ymax = 3.5)) # hide
```

```@docs
Beta
BetaPrime
Biweight
```
```@docs
Cauchy
```
Density `Cauchy(-2,1)`
```@example cauchy
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Cauchy(-2,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(xmin = -10, xmax = 10, ymax = 0.36)) #hide
```

```@docs
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
```
```@docs
LogNormal
```
Density `LogNormal(0,1)`
```@example lognormal
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = LogNormal(0, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line, Coord.cartesian(xmin = 0, xmax = 5, ymax = 0.7)) # hide
```
```@docs
NoncentralBeta
NoncentralChisq
NoncentralF
NoncentralT
Normal
NormalCanon
NormalInverseGaussian
Pareto
PGeneralizedGaussian
```
```@docs
Rayleigh
```
Density distribution `Rayleigh(0.5)`
```@example rayleigh
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Rayleigh(0.5) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(xmin = -1, xmax = 5, ymax = 1.2)) # hide
```
```@docs
Semicircle
StudentizedRange
SymTriangularDist
TDist
TriangularDist
Triweight
Uniform
VonMises
```
```@docs
Weibull
```
Density plot `Weibull(0.5,1)`
```@example weibull
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Weibull(0.5, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line,Coord.cartesian(xmin = 0, xmax = 2, ymax = 2.0)) # hide
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
