# [Univariate Distributions](@id univariates)

*Univariate distributions* are the distributions whose variate forms are `Univariate` (*i.e* each sample is a scalar). Abstract types for univariate distributions:

```julia
const UnivariateDistribution{S<:ValueSupport} = Distribution{Univariate,S}

const DiscreteUnivariateDistribution   = Distribution{Univariate, Discrete}
const ContinuousUnivariateDistribution = Distribution{Univariate, Continuous}
```

## Common Interface

A series of methods is implemented for each univariate distribution, which provides
useful functionalities such as moment computation, pdf evaluation, and sampling
(*i.e.* random number generation).

### Parameter Retrieval

**Note:** `params` are defined for all univariate distributions, while other parameter
retrieval methods are only defined for those distributions for which these parameters make sense.
See below for details.

```@docs
params(::UnivariateDistribution)
partype
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
entropy(::UnivariateDistribution, ::Real)
mgf(::UnivariateDistribution, ::Any)
cgf(::UnivariateDistribution, ::Any)
cf(::UnivariateDistribution, ::Any)
pdfsquaredL2norm
```

### Probability Evaluation

```@docs
insupport(::UnivariateDistribution, x::Any)
pdf(::UnivariateDistribution, ::Real)
logpdf(::UnivariateDistribution, ::Real)
gradlogpdf(::ContinuousUnivariateDistribution, ::Real)
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

```@setup plotdensity
using Distributions
import GR

# display figures as SVGs
GR.inline("svg")

# generic plot density of univariate distribution
function _plotdensity(
    xs,
    dist::UnivariateDistribution;
    kind,
    kwargs...
)
    GR.figure(;
        grid=false,
        backgroundcolor=0, # white instead of transparent background for dark Documenter scheme
        font="Helvetica_Regular", # work around https://github.com/JuliaPlots/Plots.jl/issues/2596
        kwargs...
    )
    return kind(xs, Base.Fix1(pdf, dist))
end

# plot probability density of continuous distributions
function plotdensity(
    (xmin, xmax),
    dist::ContinuousUnivariateDistribution;
    npoints=299,
    linewidth=2.0, # thick lines
    kwargs...,
)
    _plotdensity(
        range(xmin, xmax; length=npoints),
        dist;
        kind=GR.plot,
        xlabel="x",
        ylabel="density",
        linewidth,
        kwargs...
    )
end

# plot probability density of discrete distributions
function plotdensity(
    points,
    dist::DiscreteUnivariateDistribution;
    kwargs...,
)
    _plotdensity(
        points,
        dist;
        kind=GR.stem,
        xlabel="k",
        ylabel="P(X=k)",
        kwargs...
    )
end

# convenience function with automatic title
function plotdensity(
    xmin_xmax,
    ::Type{T},
    args=();
    title=string(T) * "(" * join(args, ", ") * ")",
    kwargs...
) where {T<:UnivariateDistribution}
    return plotdensity(xmin_xmax, T(args...); title=title, kwargs...)
end
```

```@docs
Arcsine
```
```@example plotdensity
plotdensity((0.001, 0.999), Arcsine, (0, 1)) # hide
```

```@docs
Beta
```
```@example plotdensity
plotdensity((0, 1), Beta, (2, 2)) # hide
```

```@docs
BetaPrime
```
```@example plotdensity
plotdensity((0, 1), BetaPrime, (1, 2)) # hide
```

```@docs
Biweight
```
```@example plotdensity
plotdensity((-1, 3), Biweight, (1, 2)) # hide
```

```@docs
Cauchy
```
```@example plotdensity
plotdensity((-12, 5), Cauchy, (-2, 1)) # hide
```

```@docs
Chernoff
```
```@example plotdensity
plotdensity((-3, 3), Chernoff) # hide
```

```@docs
Chi
```
```@example plotdensity
plotdensity((0.001, 3), Chi, (1,)) # hide
```

```@docs
Chisq
```
```@example plotdensity
plotdensity((0, 9), Chisq, (3,)) # hide
```

```@docs
Cosine
```
```@example plotdensity
plotdensity((-1, 1), Cosine, (0, 1)) # hide
```

```@docs
Epanechnikov
```
```@example plotdensity
plotdensity((-1, 1), Epanechnikov, (0, 1)) # hide
```

```@docs
Erlang
```
```@example plotdensity
plotdensity((0, 8), Erlang, (7, 0.5)) # hide
```

```@docs
Exponential
```
```@example plotdensity
plotdensity((0, 3.5), Exponential, (0.5,)) # hide
```

```@docs
FDist
```
```@example plotdensity
plotdensity((0, 10), FDist, (10, 1)) # hide
```

```@docs
Frechet
```
```@example plotdensity
plotdensity((0, 20), Frechet, (1, 1)) # hide
```

```@docs
Gamma
```
```@example plotdensity
plotdensity((0, 18), Gamma, (7.5, 1)) # hide
```

```@docs
GeneralizedExtremeValue
```
```@example plotdensity
plotdensity((0, 30), GeneralizedExtremeValue, (0, 1, 1)) # hide
```

```@docs
GeneralizedPareto
```
```@example plotdensity
plotdensity((0, 20), GeneralizedPareto, (0, 1, 1)) # hide
```

```@docs
Gumbel
```
```@example plotdensity
plotdensity((-2, 5), Gumbel, (0, 1)) # hide
```

```@docs
InverseGamma
```
```@example plotdensity
plotdensity((0.001, 1), InverseGamma, (3, 0.5)) # hide
```

```@docs
InverseGaussian
```
```@example plotdensity
plotdensity((0, 5), InverseGaussian, (1, 1)) # hide
```

```@docs
JohnsonSU
```
```@example plotdensity
plotdensity((-20, 20), JohnsonSU, (0.0, 1.0, 0.0, 1.0)) # hide
```

```@docs
Kolmogorov
```
```@example plotdensity
plotdensity((0, 2), Kolmogorov) # hide
```

```@docs
KSDist
KSOneSided
```

```@docs
Kumaraswamy
```
```@example plotdensity
plotdensity((0, 1), Kumaraswamy, (2, 5)) # hide
```

```@docs
Laplace
```
```@example plotdensity
plotdensity((-20, 20), Laplace, (0, 4)) # hide
```

```@docs
Levy
```
```@example plotdensity
plotdensity((0, 20), Levy, (0, 1)) # hide
```

```@docs
Lindley
```
```@example plotdensity
plotdensity((0, 20), Lindley, (1.5,)) # hide
```

```@docs
Logistic
```
```@example plotdensity
plotdensity((-4, 8), Logistic, (2, 1)) # hide
```

```@docs
LogLogistic
```
```@example plotdensity
plotdensity((0, 2), LogLogistic, (2, 1)) # hide
```

```@docs
LogitNormal
```
```@example plotdensity
plotdensity((0, 1), LogitNormal, (0, 1)) # hide
```

```@docs
LogNormal
```
```@example plotdensity
plotdensity((0, 5), LogNormal, (0, 1)) # hide
```

```@docs
LogUniform
```
```@example plotdensity
plotdensity((0, 11), LogUniform, (1, 10)) # hide
```

```@docs
NoncentralBeta
```
```@example plotdensity
plotdensity((0, 1), NoncentralBeta, (2, 3, 1)) # hide
```

```@docs
NoncentralChisq
```
```@example plotdensity
plotdensity((0, 20), NoncentralChisq, (2, 3)) # hide
```

```@docs
NoncentralF
```
```@example plotdensity
plotdensity((0, 10), NoncentralF, (2, 3, 1)) # hide
```

```@docs
NoncentralT
```
```@example plotdensity
plotdensity((-1, 20), NoncentralT, (2, 3)) # hide
```

```@docs
Normal
```
```@example plotdensity
plotdensity((-4, 4), Normal, (0, 1)) # hide
```

```@docs
NormalCanon
```
```@example plotdensity
plotdensity((-4, 4), NormalCanon, (0, 1)) # hide
```

```@docs
NormalInverseGaussian
```
```@example plotdensity
plotdensity((-2, 2), NormalInverseGaussian, (0, 0.5, 0.2, 0.1)) # hide
```

```@docs
Pareto
```
```@example plotdensity
plotdensity((1, 8), Pareto, (1, 1)) # hide
```

```@docs
PGeneralizedGaussian
```
```@example plotdensity
plotdensity((0, 20), PGeneralizedGaussian, (0.2)) # hide
```

```@docs
Rayleigh
```
```@example plotdensity
plotdensity((0, 2), Rayleigh, (0.5)) # hide
```

```@docs
Rician
```
```@example plotdensity
plotdensity((0, 5), Rician, (0.5, 1)) # hide
```

```@docs
Semicircle
```
```@example plotdensity
plotdensity((-1, 1), Semicircle, (1,)) # hide
```

```@docs
SkewedExponentialPower
```
```@example plotdensity
plotdensity((-8, 5), SkewedExponentialPower, (0, 1, 0.7, 0.7)) # hide
```

```@docs
SkewNormal
```
```@example plotdensity
plotdensity((-4, 4), SkewNormal, (0, 1, -1)) # hide
```

```@docs
StudentizedRange
SymTriangularDist
```
```@example plotdensity
# we only need to plot 5 equally spaced points for these parameters and limits # hide
plotdensity((-2, 2), SymTriangularDist, (0, 1); npoints=5) # hide
```

```@docs
TDist
```
```@example plotdensity
plotdensity((-5, 5), TDist, (5,)) # hide
```

```@docs
TriangularDist
```
```@example plotdensity
# we only need to plot 6 equally spaced points for these parameters and limits # hide
plotdensity((-0.5, 2), TriangularDist, (0, 1.5, 0.5); npoints=6) # hide
```

```@docs
Triweight
```
```@example plotdensity
plotdensity((0, 2), Triweight, (1, 1)) # hide
```

```@docs
Uniform
```
```@example plotdensity
plotdensity((-0.5, 1.5), Uniform, (0, 1); ylim=(0, 1.5)) # hide
```

```@docs
VonMises
```
```@example plotdensity
plotdensity((-π, π), VonMises, (0.5,); xlim=(-π, π), xticks=(π/5, 5), xticklabels=x -> x ≈ -π ? "-π" : (x ≈ π ? "π" : "0")) # hide
```

```@docs
Weibull
```
```@example plotdensity
plotdensity((0.001, 3), Weibull, (0.5, 1)) # hide
```

## Discrete Distributions

```@docs
Bernoulli
```
```@example plotdensity
plotdensity([0, 1], Bernoulli, (0.3,); xlim=(-1, 2), ylim=(0, 1)) # hide
```

```@docs
BernoulliLogit
```
```@example plotdensity
plotdensity([0, 1], BernoulliLogit, (0.3,); xlim=(-1, 2), ylim=(0, 1)) # hide
```

```@docs
BetaBinomial
```
```@example plotdensity
plotdensity(0:10, BetaBinomial, (10, 2, 3), xlim=(-1, 11), ylim=(0, 0.16)) # hide
```

```@docs
Binomial
```
```@example plotdensity
plotdensity(0:20, Binomial, (20, 0.3), xlim=(-1, 20), ylim=(0, 0.2)) # hide
```

```@docs
Categorical
```
```@example plotdensity
# need specific title because string(Categorical) = "Categorical{P} where P<:Real" instead of just "Categorical" # hide
ps = [0.3, 0.2, 0.5] # hide
plotdensity(1:3, Categorical, (ps,), xlim=(0, 4), ylim=(0, 0.7), title="Categorical($ps)") # hide
```

```@docs
Dirac
```
```@example plotdensity
plotdensity([2], Dirac, (2,); xlim=(0, 4), ylim=(0, 1.3)) # hide
```

```@docs
DiscreteUniform
```
```@example plotdensity
plotdensity(1:6, DiscreteUniform, (1, 6); xlim=(0, 7), ylim=(0, 0.28), title_size=20) # hide
```

```@docs
DiscreteNonParametric
```
```@example plotdensity
plotdensity([1, 4, 6, 9], DiscreteNonParametric, ([1, 4, 6, 9], [0.3, 0.2, 0.1, 0.4]); xlim=(0, 10), ylim=(0, 0.5)) # hide
```

```@docs
Geometric
```
```@example plotdensity
plotdensity(0:15, Geometric, (0.3,); xlim=(-1, 15), ylim=(0, 0.35)) # hide
```

```@docs
Hypergeometric
```
```@example plotdensity
plotdensity(10:20, Hypergeometric, (20, 10, 20); xlim=(9, 21), ylim=(0, 0.4)) # hide
```

```@docs
NegativeBinomial
```
```@example plotdensity
plotdensity(0:30, NegativeBinomial, (7, 0.5), xlim=(-1, 30), ylim=(0, 0.13)) # hide
```

```@docs
Poisson
```
```@example plotdensity
plotdensity(0:20, Poisson, (5,); xlim=(-1, 20), ylim=(0, 0.2)) # hide
```

```@docs
PoissonBinomial
```
```@example plotdensity
plotdensity(0:7, PoissonBinomial, ([0.1, 0.7, 0.4, 0.5, 0.3, 0.8, 0.5],); xlim=(-1, 8), ylim=(0.0, 0.35)) # hide
```

```@docs
Skellam
```
```@example plotdensity
plotdensity(-20:20, Skellam, (8, 5), xlim=(-20, 20), ylim=(0, 0.14)) # hide
```

```@docs
Soliton
```
```@example plotdensity
plotdensity(1:20, Soliton, (20, 10, 0.1), xlim=(0, 21), ylim=(0, 0.4)) # hide
```

### Vectorized evaluation

Vectorized computation and in-place vectorized computation have been deprecated.

## Index

```@index
Pages = ["univariate.md"]
```
