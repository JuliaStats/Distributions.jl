"""
    AffineDistribution(μ, σ, ρ)

A shifted and scaled (affinely transformed) version of `ρ`.

If ``Z`` is a random variable with distribution `ρ`, then `AffineDistribution(μ, σ, ρ)` is
the distribution of the random variable
```math
X = μ + σ Z
```

If `ρ` is a discrete univariate distribution, the probability mass function of the
transformed distribution is given by
```math
P(X = x) = P\\left(Z = \\frac{x-μ}{σ} \\right).
```

If `ρ` is a continuous univariate distribution with probability density function ``f_Z``,
the probability density function of the transformed distribution is given by
```math
f_X(x) = \\frac{1}{|σ|} f_Z\\left( \\frac{x-μ}{σ} \\right).
```

We recommend against using the `AffineDistribution` constructor directly. Instead, use
`+`, `-`, `*`, and `/`. These are optimized for specific distributions and will fall back
on `AffineDistribution` only when they need to.

Affine transformations of discrete variables are easily affected by rounding errors. If you
are getting incorrect results, try using exact `Rational` types instead of floats.

```julia
d = μ + σ * ρ       # Create location-scale transformed distribution
params(d)           # Get the parameters, i.e. (μ, σ, ρ)
```
"""
struct AffineDistribution{T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}} <: UnivariateDistribution{S}
    μ::T
    σ::T
    ρ::D
    # TODO: Remove? It is not used in Distributions anymore
    function AffineDistribution{T,S,D}(μ::T, σ::T, ρ::D; check_args::Bool=true) where {T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}}
        @check_args AffineDistribution (σ, !iszero(σ))
        new{T, S, D}(μ, σ, ρ)
    end
    function AffineDistribution{T}(μ::T, σ::T, ρ::UnivariateDistribution) where {T<:Real}
        D = typeof(ρ)
        S = value_support(D)
        return new{T,S,D}(μ, σ, ρ)
    end
end

function AffineDistribution(μ::T, σ::T, ρ::UnivariateDistribution; check_args::Bool=true) where {T<:Real}
    @check_args AffineDistribution (σ, !iszero(σ))
    _T = promote_type(eltype(ρ), T)
    return AffineDistribution{_T}(_T(μ), _T(σ), ρ)
end

function AffineDistribution(μ::Real, σ::Real, ρ::UnivariateDistribution; check_args::Bool=true)
    return AffineDistribution(promote(μ, σ)..., ρ; check_args=check_args)
end

# aliases
const LocationScale{T,S,D} = AffineDistribution{T,S,D}
function LocationScale(μ::Real, σ::Real, ρ::UnivariateDistribution; check_args::Bool=true)
    Base.depwarn("`LocationScale` is deprecated. Use `+` and `*` instead", :LocationScale)
    # preparation for future PR where I remove σ > 0 check
    @check_args LocationScale (σ, σ > zero(σ))
    return AffineDistribution(μ, σ, ρ; check_args=false)
end

const ContinuousAffineDistribution{T<:Real,D<:ContinuousUnivariateDistribution} = AffineDistribution{T,Continuous,D}
const DiscreteAffineDistribution{T<:Real,D<:DiscreteUnivariateDistribution} = AffineDistribution{T,Discrete,D}

Base.eltype(::Type{<:AffineDistribution{T}}) where T = T

minimum(d::AffineDistribution) =
    d.σ > 0 ? d.μ + d.σ * minimum(d.ρ) : d.μ + d.σ * maximum(d.ρ)
maximum(d::AffineDistribution) =
    d.σ > 0 ? d.μ + d.σ * maximum(d.ρ) : d.μ + d.σ * minimum(d.ρ)
support(d::AffineDistribution) = affinedistribution_support(d.μ, d.σ, support(d.ρ))
function affinedistribution_support(μ::Real, σ::Real, support::RealInterval)
    if σ > 0
        return RealInterval(μ + σ * support.lb, μ + σ * support.ub)
    else
        return RealInterval(μ + σ * support.ub, μ + σ * support.lb)
    end
end
affinedistribution_support(μ::Real, σ::Real, support) = σ > 0 ? μ .+ σ .* support : μ .+ σ .* reverse(support)

AffineDistribution(μ::Real, σ::Real, d::AffineDistribution) = AffineDistribution(μ + d.μ * σ, σ * d.σ, d.ρ)

#### Conversions

convert(::Type{AffineDistribution{T}}, μ::Real, σ::Real, ρ::D) where {T<:Real, D<:UnivariateDistribution} = AffineDistribution(T(μ),T(σ),ρ)
function Base.convert(::Type{AffineDistribution{T}}, d::AffineDistribution) where {T<:Real}
    AffineDistribution{T}(T(d.μ), T(d.σ), d.ρ)
end
Base.convert(::Type{AffineDistribution{T}}, d::AffineDistribution{T}) where {T<:Real} = d

#### Parameters

location(d::AffineDistribution) = d.μ
scale(d::AffineDistribution) = d.σ
params(d::AffineDistribution) = (d.μ,d.σ,d.ρ)
partype(::AffineDistribution{T}) where {T} = T

#### Statistics

mean(d::AffineDistribution) = d.μ + d.σ * mean(d.ρ)
median(d::AffineDistribution) = d.μ + d.σ * median(d.ρ)
mode(d::AffineDistribution) = d.μ + d.σ * mode(d.ρ)
modes(d::AffineDistribution) = d.μ .+ d.σ .* modes(d.ρ)

var(d::AffineDistribution) = d.σ^2 * var(d.ρ)
std(d::AffineDistribution) = abs(d.σ) * std(d.ρ)
skewness(d::AffineDistribution) = sign(d.σ) * skewness(d.ρ)
kurtosis(d::AffineDistribution) = kurtosis(d.ρ)

isplatykurtic(d::AffineDistribution) = isplatykurtic(d.ρ)
isleptokurtic(d::AffineDistribution) = isleptokurtic(d.ρ)
ismesokurtic(d::AffineDistribution) = ismesokurtic(d.ρ)

entropy(d::ContinuousAffineDistribution) = entropy(d.ρ) + log(abs(d.σ))
entropy(d::DiscreteAffineDistribution) = entropy(d.ρ)

mgf(d::AffineDistribution,t::Real) = exp(d.μ*t) * mgf(d.ρ,d.σ*t)

#### Evaluation & Sampling

pdf(d::ContinuousAffineDistribution, x::Real) = pdf(d.ρ,(x-d.μ)/d.σ) / abs(d.σ)
pdf(d::DiscreteAffineDistribution, x::Real) = pdf(d.ρ,(x-d.μ)/d.σ)

logpdf(d::ContinuousAffineDistribution,x::Real) = logpdf(d.ρ,(x-d.μ)/d.σ) - log(abs(d.σ))
logpdf(d::DiscreteAffineDistribution, x::Real) = logpdf(d.ρ,(x-d.μ)/d.σ)

# CDF methods

for (f, fc) in ((:cdf, :ccdf), (:ccdf, :cdf), (:logcdf, :logccdf), (:logccdf, :logcdf))
    @eval function $f(d::ContinuousAffineDistribution, x::Real)
        z = (x - d.μ) / d.σ
        return d.σ > 0 ? $f(d.ρ, z) : $fc(d.ρ, z)
    end
end

function cdf(d::DiscreteAffineDistribution, x::Real)
    z = (x - d.μ) / d.σ
    # Have to include probability mass at endpoints
    return d.σ > 0 ? cdf(d.ρ, z) : (ccdf(d.ρ, z) + pdf(d.ρ, z))
end
function ccdf(d::DiscreteAffineDistribution, x::Real)
    z = (x - d.μ) / d.σ
    # Have to exclude probability mass at endpoints
    return d.σ > 0 ? ccdf(d.ρ, z) : (cdf(d.ρ, z) - pdf(d.ρ, z))
end
function logcdf(d::DiscreteAffineDistribution, x::Real)
    z = (x - d.μ) / d.σ
    return d.σ > 0 ? logcdf(d.ρ, z) : logaddexp(logccdf(d.ρ, z), logpdf(d.ρ, z))
end
function logccdf(d::DiscreteAffineDistribution, x::Real)
    z = (x - d.μ) / d.σ
    return d.σ > 0 ? logccdf(d.ρ, z) : logsubexp(logcdf(d.ρ, z), logpdf(d.ρ, z))
end

quantile(d::AffineDistribution, q::Real) = d.μ + d.σ * quantile(d.ρ, d.σ > 0 ? q : 1 - q)

rand(rng::AbstractRNG, d::AffineDistribution) = d.μ + d.σ * rand(rng, d.ρ)
cf(d::AffineDistribution, t::Real) = cf(d.ρ,t*d.σ) * exp(1im*t*d.μ)
gradlogpdf(d::ContinuousAffineDistribution, x::Real) = gradlogpdf(d.ρ,(x-d.μ)/d.σ) / d.σ

#### Syntactic sugar for simple transforms of distributions, e.g., d + x, d - x, and so on

Base.:+(d::UnivariateDistribution, x::Real) = AffineDistribution(x, one(x), d)
Base.:+(x::Real, d::UnivariateDistribution) = d + x
Base.:*(x::Real, d::UnivariateDistribution) = AffineDistribution(zero(x), x, d)
Base.:*(d::UnivariateDistribution, x::Real) = x * d
Base.:-(d::UnivariateDistribution, x::Real) = d + -x
Base.:-(d::UnivariateDistribution) = -one(partype(d)) * d
Base.:/(d::UnivariateDistribution, x::Real) = inv(x) * d
