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
Density `Arcsine(0,1)`

```@example Arcsine
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Arcsine(0,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(ymin = 0, ymax = 3.5)) # hide
```

```@docs
Beta
```
Density `Beta(2,2)`

```@example Beta
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Beta(2,2) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(ymin = 0, ymax = 2.2)) # hide
```

```@docs
BetaPrime
```
Density `BetaPrime(1,2)`
```@example BetaPrime
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = BetaPrime(1,2) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(ymin = 0, ymax = 2)) # hide
```

```@docs
Biweight
```
Density `Biweight(1,2)`

```@example Biweight
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Biweight(1,2) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(ymin = 0, ymax = 0.5)) # hide
```
```@docs
Cauchy
```
Density `Cauchy(-2,1)`
```@example cauchy
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Cauchy(-2,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(xmin = -10, xmax = 10, ymax = 0.36)) #hide
```

```@docs
Chernoff
```

```@docs
Chi
```
Density `Chi(1)`
```@example Chi
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Chi(1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.8)) #hide
```

```@docs
Chisq
```
Density `Chisq(3)`
```@example chisq
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Chisq(3) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.27)) #hide
```

```@docs
Cosine
```
Density `Cosine(0,1)`
```@example Cosine
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Cosine(0, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 1)) #hide
```

```@docs
Epanechnikov
```
Density `Epanechnikov(0,1)`
```@example Cosine
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Epanechnikov(0, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 1)) #hide
```

```@docs
Erlang
```
Density `Erlang(7,0.5)`
```@example Erlang
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Erlang(7, 0.5) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.4)) #hide
```

```@docs
Exponential
```
Density `Exponential(0.5)`
```@example Exponential
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Exponential(0.5) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.55)) #hide
```

```@docs
FDist
```
Density `FDist(10,1)`
```@example FDist
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = FDist(10,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(xmin = 0, xmax = 10, ymin = 0, ymax = 0.55)) #hide
```

```@docs
Frechet
```
Density `Frechet(1,1)`
```@example Frechet
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Frechet(1,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x),  Geom.line, Coord.cartesian(xmin = 0, xmax = 50,ymin = 0, ymax = 0.55)) #hide
```

```@docs
Gamma
```
Density `Gamma(7.5, 1.0)`
```@example Gamma
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Gamma(7.5, 1.0) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.16)) #hide
```

```@docs
GeneralizedExtremeValue
```
Density `GeneralizedExtremeValue(0,1,1)`
```@example GeneralizedExtremeValue
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = GeneralizedExtremeValue(0,1,1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(xmin = 0, xmax = 50, ymin = 0, ymax = 0.16)) #hide
```

```@docs
GeneralizedPareto
```
Density `GeneralizedPareto(0,1,1)`
```@example GeneralizedPareto
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = GeneralizedPareto(0,1,1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.16)) #hide
```

```@docs
Gumbel
```
Density `Gumbel(0,1)`
```@example Gumbel
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Gumbel(0,1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.5)) #hide
```

```@docs
InverseGamma
```
Density `InverseGamma(3,0.5)`
```@example InverseGamma
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = InverseGamma(3,0.5) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 5)) #hide
```

```@docs
InverseGaussian
```
Density `InverseGaussian(1,1)`
```@example InverseGaussian
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = InverseGaussian(1,1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 1.3)) #hide
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
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Laplace(0,4) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.15)) #hide
```

```@docs
Levy
```
Density `Levy(0,1)`
```@example Levy
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Levy(0,1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(xmin = 0, xmax = 20, ymin = 0, ymax = 0.6)) #hide
```

```@docs
LocationScale
```
Density `LocationScale(2,1,Normal(0,1))`
```@example LocationScale
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d1 = Normal(0,1) # hide
d = LocationScale(2.0,1.0, d1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.5)) #hide
```

```@docs
Logistic
```
Density `Logistic(2,1)`
```@example Logistic
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
set_default_plot_size(10cm, 10cm) # hide
d = Logistic(2,1) # hide
x = rand(d, 100) # hide
plot(x = x,y = pdf.(d,x),  Geom.line, Coord.cartesian(ymin = 0, ymax = 0.25)) #hide
```

```@docs
LogitNormal
```
Density `LogitNormal(0,1)`
```@example LogitNormal
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = LogitNormal(1.78) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line, Coord.cartesian(xmin = 0, xmax = 1, ymax = 4.8)) # hide
```

```@docs
LogNormal
```
Density `LogNormal(0,1)`
```@example LogNormal
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = LogNormal(0, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line, Coord.cartesian(xmin = 0, xmax = 10, ymax = 0.7)) # hide
```

```@docs
NoncentralBeta
```
Density `NoncentralBeta(2,3,1)`
```@example NoncentralBeta
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = NoncentralBeta(0.5,1,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line) # hide
```

```@docs
NoncentralChisq
```
Density `NoncentralChisq(0.5,1,1)`
```@example NoncentralChisq
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = NoncentralChisq(2,3) # hide
x = rand(d, 1000) # hide
plot(x = x, Geom.density) #hide
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
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Normal(0, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line, Coord.cartesian(xmin = -4, xmax = 4, ymax = 0.42)) # hide
```

```@docs
NormalCanon
```

```@docs
NormalInverseGaussian
```
Density `NormalInverseGaussian(0,0.5,0.2,0.1)`
```@example NormalInverseGaussian
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = NormalInverseGaussian(0,0.5,0.2,0.1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line, Coord.cartesian(xmin = -4, xmax = 4, ymax = 3.5)) # hide
```

```@docs
Pareto
```
Density `Pareto(1,1)`
```@example Pareto
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Pareto(1,1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d,x), Geom.line, Coord.cartesian(xmin = 0, xmax = 8, ymax = 1)) # hide
```

```@docs
PGeneralizedGaussian
```
Density `PGeneralizedGaussian(0.2)`
```@example Pareto
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = PGeneralizedGaussian(0.2) # hide
x = rand(d, 1000) # hide
plot(x = x, Geom.density) #hide
```

```@docs
Rayleigh
```
Density `Rayleigh(0.5)`
```@example Rayleigh
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Rayleigh(0.5) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(xmin = 0, xmax = 1.8, ymax = 1.2)) # hide
```
```@docs
Semicircle
```
Density `Semicircle(1)`
```@example Semicircle
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Semicircle(1) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```

```@docs
StudentizedRange
```
Density `StudentizedRange(5,2)`
```@example Semicircle
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = StudentizedRange(5,2) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```

```@docs
SymTriangularDist
```
Density `SymTriangularDist(0,1)`
```@example SymTriangularDist
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = SymTriangularDist(0,1) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```

```@docs
TDist
```
Density `TDist(5)`
```@example TDist
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = TDist(5) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```
```@docs
TriangularDist
```
Density `TriangularDist(1,1)`
```@example TriangularDist
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = TriangularDist(1,1) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```

```@docs
Triweight
```
Density `Triweight(1,1)`
```@example Triweight
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Triweight(1,1) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```
```@docs
Uniform
```
Density `Uniform(0,1)`
```@example Uniform
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Uniform(0,1) # hide
x = rand(d, 10000) # hide
plot(x = x, Geom.density) # hide
```

```@docs
VonMises
```
Density `VonMises(0.5)`
```@example VonMises
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = VonMises(0.5) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(ymin = 0, ymax = 0.25)) # hide
```

```@docs
Weibull
```
Density `Weibull(0.5,1)`
```@example Weibull
using Random, Distributions, Gadfly # hide
Random.seed!(123) # hide
d = Weibull(0.5, 1) # hide
x = rand(d, 100) # hide
plot(x = x, y = pdf.(d, x), Geom.line, Coord.cartesian(xmin = 0, xmax = 2, ymax = 2.0)) # hide
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
