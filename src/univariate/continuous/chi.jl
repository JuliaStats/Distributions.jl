doc"""
    Chi(ν)

The *Chi distribution* `ν` degrees of freedom has probability density function

$f(x; k) = \frac{1}{\Gamma(k/2)} 2^{1 - k/2} x^{k-1} e^{-x^2/2}, \quad x > 0$

It is the distribution of the square-root of a [`Chisq`](:func:`Chisq`) variate.

```julia
Chi(k)       # Chi distribution with k degrees of freedom

params(d)    # Get the parameters, i.e. (k,)
dof(d)       # Get the degrees of freedom, i.e. k
```

External links

* [Chi distribution on Wikipedia](http://en.wikipedia.org/wiki/Chi_distribution)

"""
immutable Chi <: ContinuousUnivariateDistribution
    ν::Float64

    Chi(ν::Real) = (@check_args(Chi, ν > zero(ν)); new(ν))
end

@distr_support Chi 0.0 Inf

#### Parameters

dof(d::Chi) = d.ν
params(d::Chi) = (d.ν,)


#### Statistics

mean(d::Chi) = (h = d.ν * 0.5; sqrt2 * gamma(h + 0.5) / gamma(h))

var(d::Chi) = d.ν - mean(d)^2
_chi_skewness(μ::Float64, σ::Float64) = (σ2 = σ^2; σ3 = σ2 * σ; (μ / σ3) * (1.0 - 2.0 * σ2))

function skewness(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    _chi_skewness(μ, σ)
end

function kurtosis(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    γ = _chi_skewness(μ, σ)
    (2.0 / σ^2) * (1 - μ * σ * γ - σ^2)
end

entropy(d::Chi) = (ν = d.ν;
    lgamma(ν / 2.0) - 0.5 * logtwo - ((ν - 1.0) / 2.0) * digamma(ν / 2.0) + ν / 2.0)

function mode(d::Chi)
    d.ν >= 1.0 || error("Chi distribution has no mode when ν < 1")
    sqrt(d.ν - 1.0)
end


#### Evaluation

pdf(d::Chi, x::Float64) = exp(logpdf(d, x))

logpdf(d::Chi, x::Float64) = (ν = d.ν;
    (1.0 - 0.5 * ν) * logtwo + (ν - 1.0) * log(x) - 0.5 * x^2 - lgamma(0.5 * ν)
)

gradlogpdf(d::Chi, x::Float64) = x >= 0.0 ? (d.ν - 1.0) / x - x : 0.0

cdf(d::Chi, x::Float64) = chisqcdf(d.ν, x^2)
ccdf(d::Chi, x::Float64) = chisqccdf(d.ν, x^2)
logcdf(d::Chi, x::Float64) = chisqlogcdf(d.ν, x^2)
logccdf(d::Chi, x::Float64) = chisqlogccdf(d.ν, x^2)

quantile(d::Chi, p::Float64) = sqrt(chisqinvcdf(d.ν, p))
cquantile(d::Chi, p::Float64) = sqrt(chisqinvccdf(d.ν, p))
invlogcdf(d::Chi, p::Float64) = sqrt(chisqinvlogcdf(d.ν, p))
invlogccdf(d::Chi, p::Float64) = sqrt(chisqinvlogccdf(d.ν, p))


#### Sampling

rand(d::Chi) = sqrt(_chisq_rand(d.ν))
