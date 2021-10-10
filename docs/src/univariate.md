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
pdfsquaredL2norm
```

### Probability Evaluation

```@docs
insupport(::UnivariateDistribution, x::Any)
pdf(::UnivariateDistribution, ::Real)
logpdf(::UnivariateDistribution, ::Real)
loglikelihood(::UnivariateDistribution, ::AbstractArray)
cdf(::UnivariateDistribution, ::Real)
logcdf(::UnivariateDistribution, ::Real)
logdiffcdf(::UnivariateDistribution, ::Real, ::Real)
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
Density `Arcsine(0,1)`

```@example Arcsine
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Arcsine(0,1) # hide
xgrid = 0:0.001:1 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

```@docs
Beta
```
Density `Beta(2,2)`

```@example Beta
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Beta(2,2) # hide
xgrid = 0:0.001:1  # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

```@docs
BetaPrime
```
Density `BetaPrime(1,2)`
```@example BetaPrime
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = BetaPrime(1,2) # hide
xgrid = 0:0.001:1 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

```@docs
Biweight
```
Density `Biweight(1,2)`

```@example Biweight
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Biweight(1,2) # hide
xgrid = -1:0.001:3 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```
```@docs
Cauchy
```
Density `Cauchy(-2,1)`
```@example cauchy
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Cauchy(-2,1) # hide
xgrid = -10.5:0.001:5 # hide
plot(x = xgrid, y = pdf.(d,xgrid),  Geom.line) #hide
```

```@docs
Chernoff
```

```@docs
Chi
```
Density `Chi(1)`
```@example Chi
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Chi(1) # hide
xgrid = 0:0.001:3 # hide
plot(x = xgrid, y = pdf.(d,xgrid),  Geom.line) #hide
```

```@docs
Chisq
```
Density `Chisq(3)`
```@example chisq
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Chisq(3) # hide
xgrid = 0:0.1:9 # hide
plot(x = xgrid, y = pdf.(d,xgrid),  Geom.line) #hide
```

```@docs
Cosine
```
Density `Cosine(0,1)`
```@example Cosine
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Cosine(0, 1) # hide
xgrid = -1:0.001:1 # hide
plot(x = xgrid, y = pdf.(d,xgrid),  Geom.line) #hide
```

```@docs
Epanechnikov
```
Density `Epanechnikov(0,1)`
```@example Epanechnikov
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Epanechnikov(0, 1) # hide
xgrid = -1:0.001:1 # hide
plot(x = xgrid, y = pdf.(d,xgrid),  Geom.line) #hide
```

```@docs
Erlang
```
Density `Erlang(7,0.5)`
```@example Erlang
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Erlang(7, 0.5) # hide
xgrid = 0:0.001:8 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
Exponential
```
Density `Exponential(0.5)`
```@example Exponential
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Exponential(0.5) # hide
xgrid = 0:0.001:3.5 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
FDist
```
Density `FDist(10,1)`
```@example FDist
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = FDist(10,1) # hide
xgrid = 0:0.001:10 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
Frechet
```
Density `Frechet(1,1)`
```@example Frechet
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Frechet(1,1) # hide
xgrid = 0:0.001:20 # hide
plot(x = xgrid, y = pdf.(d,xgrid),  Geom.line) #hide
```

```@docs
Gamma
```
Density `Gamma(7.5, 1.0)`
```@example Gamma
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Gamma(7.5, 1.0) # hide
xgrid = 0:0.001:18 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
GeneralizedExtremeValue
```
Density `GeneralizedExtremeValue(0,1,1)`
```@example GeneralizedExtremeValue
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = GeneralizedExtremeValue(0,1,1) # hide
xgrid = 0:0.001:30 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
GeneralizedPareto
```
Density `GeneralizedPareto(0,1,1)`
```@example GeneralizedPareto
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = GeneralizedPareto(0,1,1) # hide
xgrid = 0:0.001:20 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
Gumbel
```
Density `Gumbel(0,1)`
```@example Gumbel
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Gumbel(0,1) # hide
xgrid = -2:0.001:5 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
InverseGamma
```
Density `InverseGamma(3,0.5)`
```@example InverseGamma
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = InverseGamma(3,0.5) # hide
xgrid = 0:0.001:1 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
InverseGaussian
```
Density `InverseGaussian(1,1)`
```@example InverseGaussian
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = InverseGaussian(1,1) # hide
xgrid = 0:0.001:5 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
Kolmogorov
KSDist
KSOneSided
```

```@docs
Laplace
```
Density `Laplace(0,4)`
```@example Laplace
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Laplace(0,4) # hide
xgrid = -20:0.01:20# hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
Levy
```
Density `Levy(0,1)`
```@example Levy
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Levy(0,1) # hide
xgrid = 0:0.1:20 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
LocationScale
```
Density `LocationScale(2,1,Normal(0,1))`
```@example LocationScale
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d1 = Normal(0,1) # hide
d = LocationScale(2.0,1.0, d1) # hide
xgrid = -2:0.001:5 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
Logistic
```
Density `Logistic(2,1)`
```@example Logistic
using Distributions, Gadfly # hide
set_default_plot_size(10cm, 10cm) # hide
d = Logistic(2,1) # hide
xgrid = -4:0.001:8 # hide
plot(x = xgrid,y = pdf.(d,xgrid), Geom.line) #hide
```

```@docs
LogitNormal
```
Density `LogitNormal(0,1)`
```@example LogitNormal
using Distributions, Gadfly # hide
d = LogitNormal(1.78) # hide
xgrid = 0:0.001:1 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) # hide
```

```@docs
LogNormal
```
Density `LogNormal(0,1)`
```@example LogNormal
using Distributions, Gadfly # hide
d = LogNormal(0, 1) # hide
xgrid = 0:0.001:5 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) # hide
```

```@docs
NoncentralBeta
```
Density `NoncentralBeta(2,3,1)`
```@example NoncentralBeta
using Distributions, Gadfly # hide
d = NoncentralBeta(0.5,1,1) # hide
xgrid = 0:0.001:1 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) # hide
```

```@docs
NoncentralChisq
```
Density `NoncentralChisq(0.5,1,1)`
```@example NoncentralChisq
using Distributions, Gadfly # hide
d = NoncentralChisq(2,3) # hide
xgrid = 0:0.001:20 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) #hide
```

```@docs
NoncentralF
NoncentralT
```

```@docs
Normal
```
Density `Normal(0,1)`
```@example Normal
using Distributions, Gadfly # hide
d = Normal(0, 1) # hide
xgrid = -4:0.001:4 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) # hide
```

```@docs
NormalCanon
```

```@docs
NormalInverseGaussian
```
Density `NormalInverseGaussian(0,0.5,0.2,0.1)`
```@example NormalInverseGaussian
using Distributions, Gadfly # hide
d = NormalInverseGaussian(0,0.5,0.2,0.1) # hide
xgrid = -2:0.001:2 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) # hide
```

```@docs
Pareto
```
Density `Pareto(1,1)`
```@example Pareto
using Distributions, Gadfly # hide
d = Pareto(1,1) # hide
xgrid = 1:0.001:8 # hide
plot(x = xgrid, y = pdf.(d,xgrid), Geom.line) # hide
```

```@docs
PGeneralizedGaussian
```
Density `PGeneralizedGaussian(0.2)`
```@example Pareto
using Distributions, Gadfly # hide
d = PGeneralizedGaussian(0.2) # hide
xgrid = 0:0.001:20 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) #hide
```

```@docs
Rayleigh
```
Density `Rayleigh(0.5)`
```@example Rayleigh
using Distributions, Gadfly # hide
d = Rayleigh(0.5) # hide
xgrid =  0:0.001:2# hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```
```@docs
Rician
Semicircle
```
Density `Semicircle(1)`
```@example Semicircle
using Distributions, Gadfly # hide
d = Semicircle(1) # hide
xgrid = -1:0.001:1 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

```@docs
StudentizedRange
SymTriangularDist
```
Density `SymTriangularDist(0,1)`
```@example SymTriangularDist
using Distributions, Gadfly # hide
d = SymTriangularDist(0,1) # hide
xgrid = -1:0.1:1 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

```@docs
TDist
```
Density `TDist(5)`
```@example TDist
using Distributions, Gadfly # hide
d = TDist(5) # hide
xgrid = -5:0.001:5 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```
```@docs
TriangularDist
```
Density `TriangularDist(0,1,0.5)`
```@example TriangularDist
using Distributions, Gadfly # hide
d = TriangularDist(0,1,0.5) # hide
xgrid = 0.5:0.001:1.5 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

```@docs
Triweight
```
Density `Triweight(1,1)`
```@example Triweight
using Distributions, Gadfly # hide
d = Triweight(1,1) # hide
xgrid = 0:0.001:2 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```
```@docs
Uniform
```
Density `Uniform(0,1)`
```@example Uniform
using Distributions, Gadfly # hide
d = Uniform(0,1) # hide
xgrid = 0:0.1:1 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line, Coord.cartesian(ymin = 0, ymax = 1.5)) # hide
```

```@docs
VonMises
```
Density `VonMises(0.5)` with support `[- π, π]`
```@example VonMises
using Distributions, Gadfly # hide
d = VonMises(0.5) # hide
xgrid = -pi:0.001:pi #hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line, Coord.cartesian(xmin = -pi, xmax = pi, ymin = 0,ymax = 0.35), Guide.xticks(ticks = [-π:π; ])) # hide
```

```@docs
Weibull
```
Density `Weibull(0.5,1)`
```@example Weibull
using Distributions, Gadfly # hide
d = Weibull(0.5, 1) # hide
xgrid = 0:0.001:2 # hide
plot(x = xgrid, y = pdf.(d, xgrid), Geom.line) # hide
```

## Discrete Distributions

```@docs
Bernoulli
BetaBinomial
Binomial
Categorical
Dirac
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

## Index

```@index
Pages = ["univariate.md"]
```
