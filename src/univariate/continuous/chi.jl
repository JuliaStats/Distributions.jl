"""
    Chi(ν)

The *Chi distribution* `ν` degrees of freedom has probability density function

```math
f(x; k) = \\frac{1}{\\Gamma(k/2)} 2^{1 - k/2} x^{k-1} e^{-x^2/2}, \\quad x > 0
```

It is the distribution of the square-root of a [`Chisq`](@ref) variate.

```julia
Chi(k)       # Chi distribution with k degrees of freedom

params(d)    # Get the parameters, i.e. (k,)
dof(d)       # Get the degrees of freedom, i.e. k
```

External links

* [Chi distribution on Wikipedia](http://en.wikipedia.org/wiki/Chi_distribution)

"""
struct Chi{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    Chi{T}(ν::T) where {T} = new{T}(ν)
end

function Chi(ν::T; check_args=true) where {T<:Real}
    check_args && @check_args(Chi, ν > zero(ν))
    return Chi{T}(ν)
end

Chi(ν::Integer) = Chi(float(ν))

@distr_support Chi 0.0 Inf

### Conversions
convert(::Type{Chi{T}}, ν::Real) where {T<:Real} = Chi(T(ν))
convert(::Type{Chi{T}}, d::Chi{S}) where {T <: Real, S <: Real} = Chi(T(d.ν), check_args=false)

#### Parameters

dof(d::Chi) = d.ν
params(d::Chi) = (d.ν,)
@inline partype(d::Chi{T}) where {T<:Real} = T


#### Statistics

mean(d::Chi) = (h = d.ν/2; sqrt2 * gamma(h + 1//2) / gamma(h))

var(d::Chi) = d.ν - mean(d)^2
_chi_skewness(μ::Real, σ::Real) = (σ2 = σ^2; σ3 = σ2 * σ; (μ / σ3) * (1 - 2σ2))

function skewness(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    _chi_skewness(μ, σ)
end

function kurtosis(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    γ = _chi_skewness(μ, σ)
    (2/σ^2) * (1 - μ * σ * γ - σ^2)
end

entropy(d::Chi{T}) where {T<:Real} = (ν = d.ν;
    loggamma(ν/2) - T(logtwo)/2 - ((ν - 1)/2) * digamma(ν/2) + ν/2)

function mode(d::Chi)
    d.ν >= 1 || error("Chi distribution has no mode when ν < 1")
    sqrt(d.ν - 1)
end


#### Evaluation

pdf(d::Chi, x::Real) = exp(logpdf(d, x))

logpdf(d::Chi, x::Real) = (ν = d.ν;
    (1 - ν/2) * logtwo + (ν - 1) * log(x) - x^2/2 - loggamma(ν/2)
)

gradlogpdf(d::Chi{T}, x::Real) where {T<:Real} = x >= 0 ? (d.ν - 1) / x - x : zero(T)

cdf(d::Chi, x::Real) = chisqcdf(d.ν, x^2)
ccdf(d::Chi, x::Real) = chisqccdf(d.ν, x^2)
logcdf(d::Chi, x::Real) = chisqlogcdf(d.ν, x^2)
logccdf(d::Chi, x::Real) = chisqlogccdf(d.ν, x^2)

quantile(d::Chi, p::Real) = sqrt(chisqinvcdf(d.ν, p))
cquantile(d::Chi, p::Real) = sqrt(chisqinvccdf(d.ν, p))
invlogcdf(d::Chi, p::Real) = sqrt(chisqinvlogcdf(d.ν, p))
invlogccdf(d::Chi, p::Real) = sqrt(chisqinvlogccdf(d.ν, p))


#### Sampling

rand(rng::AbstractRNG, d::Chi) =
    (ν = d.ν; sqrt(rand(rng, Gamma(ν / 2.0, 2.0one(ν)))))

struct ChiSampler{S <: Sampleable{Univariate,Continuous}} <:
    Sampleable{Univariate,Continuous}
    s::S
end

rand(rng::AbstractRNG, s::ChiSampler) = sqrt(rand(rng, s.s))

sampler(d::Chi) = ChiSampler(sampler(Chisq(d.ν)))
