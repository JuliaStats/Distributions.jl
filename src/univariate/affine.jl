log_abs_det = first ∘ logabsdet
mult_identity(x::Real) = one(x)
mult_identity(x::AbstractVector) = Diagonal(Ones(x))
add_identity(x::Real) = zero(x)
add_identity(x::AbstractArray) = Zeros(x)

"""
    AffineDistribution(μ, σ, base_dist)

A shifted and scaled (affinely transformed) version of `base_dist`.

If ``Z`` is a random variable with distribution `base_dist`, then `AffineDistribution` with
parameters `μ`, `σ`, `base_dist` is the distribution of the random variable
```math
X = μ + σ * Z
```

If `base_dist` is a discrete distribution, the probability mass function of the transformed 
distribution is given by
```math
P(X = x) = P\\left(Z = \\frac{x-μ}{σ} \\right).
```

If `base_dist` is a continuous distribution, the probability density function of
the transformed distribution is given by
```math
f(x) = \\frac{1}{σ} base_dist \\! \\left( \\frac{x-μ}{σ} \\right).
```

If `base_dist` falls under a location-scale family, `affine` will attempt to return a 
transformed distribution of type `base_dist` if an `affine` method has been implemented. 
Otherwise, it will fall back on the `AffineDistribution` wrapper type.

```julia
d = σ * (base_dist + μ)     # Create location-scale transformed distribution
params(d)                   # Get the parameters, i.e. (μ, scale, base_dist)
location(d)                 # Get the location parameter
scale(d)                    # Get the scale parameter
```
"""
struct AffineDistribution{
        F<:VariateForm,
        S<:ValueSupport,
        Tμ<:ArrayLike,
        Tσ<:ArrayLike,
        D<:Distribution{F,S}
    } <: Distribution{F,S}
    μ::Tμ
    σ::Tσ
    base_dist::D

    function AffineDistribution(
        μ::Tμ, σ::Tσ, base_dist::D; 
        check_args=true
        ) where {
            F<:VariateForm,
            S<:ValueSupport,
            D<:Distribution{F, S},
            Tμ<:ArrayLike,
            Tσ<:ArrayLike
        }
        !check_args || @check_args(AffineDistribution, !iszero(σ))
        return new{F,S,Tμ,Tσ,D}(μ, σ, base_dist)
    end
end


# Constructors

"""
    +(shift, base_dist)

Return a version of `base_dist` that has been translated by `shift`, where `shift` can be a
real number or (for a multivariate distribution) a vector.
"""
function Base.:+(μ::ArrayLike, base_dist::Distribution)
    return AffineDistribution(μ, mult_identity(μ), base_dist)
end

"""
    *(scale_factor, base_dist)

Return a version of `base_dist` that has been scaled by `scale`, where `scale` can be a real
number or (for a multivariate distribution) matrix.
"""
function Base.:*(σ::ArrayLike, base_dist::Distribution)
    return AffineDistribution(add_identity(σ), σ, base_dist)
end


function Base.:+(μ::ArrayLike, aff_dist::AffineDistribution)
    return AffineDistribution(μ + aff_dist.μ, aff_dist.σ, aff_dist.base_dist)
end

function Base.:*(σ::ArrayLike, aff_dist::AffineDistribution)
    return AffineDistribution(σ * aff_dist.μ, σ * aff_dist.σ, aff_dist.base_dist)
end


Base.:+(d::Distribution, μ::ArrayLike) = μ + d
Base.:-(d::UnivariateDistribution) = -1 * d
Base.:-(d::Distribution, μ::ArrayLike) = d + -μ
Base.:-(μ::ArrayLike, d::Distribution) = μ + -d
Base.:*(d::Distribution, σ::ArrayLike) = σ * d
Base.:/(d::Distribution, τ::ArrayLike) = inv(τ) * d


# Aliases
@deprecate LocationScale(args...; kwargs...) AffineDistribution(args...)
const ContinuousAffine{F,Tμ,Tσ,D} = AffineDistribution{F,Continuous,Tμ,Tσ,D}
const DiscreteAffine{F,Tμ,Tσ,D} = AffineDistribution{F,Discrete,Tμ,Tσ,D}

const UnivariateAffine{S,Tμ,Tσ,D} = AffineDistribution{Univariate,S,Tμ,Tσ,D}
const MultivariateAffine{F,S,Tμ,Tσ,D} = AffineDistribution{Multivariate,S,Tμ,Tσ,D}


# Support
function minimum(d::AffineDistribution)
    minim = d.σ > 0 ? minimum : maximum
    return d.μ + d.σ * minim(d.base_dist)
end
function maximum(d::AffineDistribution)
    maxim = d.σ > 0 ? maximum : minimum
    return d.μ + d.σ * maxim(d.base_dist)
end


support(d::AffineDistribution) = affine_support(d.μ, d.σ, support(d.base_dist))
affine_support(μ::Real, σ::Real, support) = μ .+ σ .* support

function affine_support(μ::Real, σ::Real, support::RealInterval) 
    if σ > 0
        lower = support.lb
        upper = support.ub
    else
        lower = support.ub
        upper = support.lb
    end
    return RealInterval(μ + σ * lower, μ + σ * upper)
end


#### Conversions

function convert(::Type{AffineDistribution{F,S,Tμ,Tσ,D}}, d::AffineDistribution) where {
        F<:VariateForm,
        S<:ValueSupport,
        D<:Distribution{F, S},
        Tμ<:ArrayLike,
        Tσ<:ArrayLike
    }
    return AffineDistribution(Tμ(d.μ), Tσ(d.σ), convert(D, d))
end


function Base.eltype(::Type{<:AffineDistribution{F,S,Tμ,Tσ,D}}) where {
        F<:VariateForm,
        S<:ValueSupport,
        D<:Distribution{F, S},
        Tμ<:ArrayLike,
        Tσ<:ArrayLike
    }
    T = Core.Compiler.return_type(Base.:*, (Tσ, eltype(D)))
    return Core.Compiler.return_type(Base.:+, (Tμ, T))
end


#### Parameters

function location(d::AffineDistribution)
    if hasmethod(location, (typeof(d.base_dist),))
        return d.μ + location(d.base_dist)
    else
        return d.μ
    end
end
function scale(d::AffineDistribution)
    if hasmethod(scale, (typeof(d.base_dist),))
        return d.σ * scale(d.base_dist)
    else
        return d.σ
    end
end
params(d::AffineDistribution) = (d.μ, d.σ, d.base_dist)
partype(::AffineDistribution{F,S,Tμ,Tσ,D}) where {F,S,Tμ,Tσ,D} = (Tμ, Tσ)


#### Statistics

mean(d::AffineDistribution) = d.μ + d.σ * mean(d.base_dist)
median(d::AffineDistribution) = d.μ + d.σ * median(d.base_dist)
mode(d::AffineDistribution) = d.μ + d.σ * mode(d.base_dist)
modes(d::AffineDistribution) = d.μ .+ d.σ .* modes(d.base_dist)

var(d::UnivariateAffine) = d.σ^2 * var(d.base_dist)
cov(d::AffineDistribution) = d.σ * cov(d.base_dist) * d.σ'
std(d::UnivariateAffine) = abs(d.σ) * std(d.base_dist)
skewness(d::AffineDistribution) = skewness(d.base_dist)
kurtosis(d::AffineDistribution) = kurtosis(d.base_dist)

isplatykurtic(d::AffineDistribution) = isplatykurtic(d.base_dist)
isleptokurtic(d::AffineDistribution) = isleptokurtic(d.base_dist)
ismesokurtic(d::AffineDistribution) = ismesokurtic(d.base_dist)

entropy(d::ContinuousAffine) = entropy(d.base_dist) + log_abs_det(d.σ)
entropy(d::DiscreteAffine) = entropy(d.base_dist)

mgf(d::AffineDistribution, t::Real) = exp(d.μ * t) * mgf(d.base_dist, d.σ * t)
cf(d::AffineDistribution, t::Real) = cf(d.base_dist, t * d.σ) * exp(1im * t * d.μ)


#### Evaluation & Sampling

pdf(d::ContinuousAffine, x::ArrayLike) = pdf(d.base_dist, (x - d.μ) / d.σ) / abs(det(d.σ))
pdf(d::DiscreteAffine, x::ArrayLike) = pdf(d.base_dist, (x - d.μ) / d.σ)

function logpdf(d::ContinuousAffine, x::Real) 
    return logpdf(d.base_dist, d.σ \ (x - d.μ)) - log_abs_det(d.σ)
end
logpdf(d::DiscreteAffine, x::Real) = logpdf(d.base_dist, (x - d.μ) / d.σ)

function cdf(d::AffineDistribution, x::Real)
    x = d.σ \ (x - d.μ)
    return d.σ < 0 ? ccdf(d.base_dist, x) : cdf(d.base_dist, x)
end

function logcdf(d::AffineDistribution, x::Real)
    x = d.σ \ (x - d.μ)
    return d.σ < 0 ? logccdf(d.base_dist, x) : logcdf(d.base_dist, x)
end 

function quantile(d::AffineDistribution, q::Real)
    q = d.σ < 0 ? (1 - q) : q
    return d.μ + d.σ * quantile(d.base_dist, q)
end

rand(rng::AbstractRNG, d::AffineDistribution) = d.μ + d.σ * rand(rng, d.base_dist)

function gradlogpdf(d::ContinuousAffine, x::Real)
    \(d.σ, gradlogpdf(d.base_dist, \(d.σ, x - d.μ)))
end
