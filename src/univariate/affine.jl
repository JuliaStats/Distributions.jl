"""
    AffineDistribution(shift, scale, base_dist)

A shifted and scaled (affinely transformed) version of `base_dist`.

If ``Z`` is a random variable with distribution `base_dist`, then `AffineDistribution` with
parameters `shift`, `scale`, `base_dist` is the distribution of the random variable
```math
X = shift + scale * Z
```

Note that `scale(x) != x.scale` in general; if `base_dist` already has a scale parameter,
`scale(x::AffineDistribution)` will return `x.scale * scale(base_dist)`.

If `base_dist` is a discrete distribution, the probability mass function of the transformed 
distribution is given by
```math
P(X = x) = P\\left(Z = \\frac{x-shift}{scale} \\right).
```

If `base_dist` is a continuous distribution, the probability density function of
the transformed distribution is given by
```math
f(x) = \\frac{1}{scale} base_dist \\! \\left( \\frac{x-shift}{scale} \\right).
```

If `base_dist` falls under a location-scale family, `affine` will attempt to return a 
transformed distribution of type `base_dist` if an `affine` method has been implemented. 
Otherwise, it will fall back on the `AffineDistribution` wrapper type.

```julia
affine(shift, scale, base_dist)  # location-scale transformed distribution
params(d)            # Get the parameters, i.e. (shift, scale, base_dist)
location(d)          # Get the location parameter
scale(d)             # Get the scale parameter
```
"""
struct AffineDistribution{T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}} <: UnivariateDistribution{S}
    shift::T
    scale::T
    base_dist::D
    function AffineDistribution{T,S,D}(shift::T, scale::T, base_dist::D; check_args=true) where {
        T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}
    }
        !check_args || @check_args(AffineDistribution, !iszero(scale))
        return new{T,S,D}(shift, scale, base_dist)
    end
end


"""
    affine(shift, scale, base_dist)

Return a shifted and scaled (affinely transformed) version of `base_dist`.

If ``Z`` is a random variable with distribution `base_dist`, then `affine` returns the 
distribution of the random variable
```math
X = shift + scale * Z
```

If `base_dist` is a discrete distribution, the probability mass function of the transformed 
distribution is given by
```math
P(X = x) = P\\left(Z = \\frac{x-shift}{scale} \\right).
```
If `base_dist` is a continuous distribution, the probability density function of
the transformed distribution is given by
```math
f(x) = \\frac{1}{scale} base_dist \\! \\left( \\frac{x-shift}{scale} \\right).
```

If `base_dist` falls under a location-scale family, `affine` will attempt to return a 
transformed distribution of type `base_dist` if an `affine` method has been implemented. 
Otherwise, it will fall back on the `AffineDistribution` wrapper type.

```julia
affine(shift, scale, base_dist)  # location-scale transformed distribution
params(d)            # Get the parameters, i.e. (shift, scale, base_dist)
location(d)          # Get the location parameter
scale(d)             # Get the scale parameter
```
"""
function affine(shift::Real, scale::Real, base_dist::UnivariateDistribution)
    T = promote_type(eltype(base_dist), typeof(shift), typeof(scale))
    D = typeof(base_dist)
    S = value_support(D)
    return AffineDistribution{T,S,D}(T(shift), T(scale), base_dist)
end

# Composing affine transformations
function affine(shift::Real, scale::Real, d::AffineDistribution)
    return affine(shift + scale * d.shift, scale * d.scale, d.base_dist)
end


# Aliases
@deprecate LocationScale(args...; kwargs...) AffineDistribution(args...)
const LocationScale = AffineDistribution
const ContinuousAffine{T<:Real,D<:ContinuousUnivariateDistribution} = AffineDistribution{T,Continuous,D}
const DiscreteAffine{T<:Real,D<:DiscreteUnivariateDistribution} = AffineDistribution{T,Discrete,D}


# Support
function minimum(d::AffineDistribution)
    minim = d.scale > 0 ? minimum : maximum
    return d.shift + d.scale * minim(d.base_dist)
end
function maximum(d::AffineDistribution)
    maxim = d.scale > 0 ? maximum : minimum
    return d.shift + d.scale * maxim(d.base_dist)
end
support(d::AffineDistribution) = affine_support(d.shift, d.scale, support(d.base_dist))

affine_support(shift::Real, scale::Real, support) = shift .+ scale .* support
function affine_support(shift::Real, scale::Real, support::RealInterval)
    if scale > 0
        lower = support.lb
        upper = support.ub
    else
        lower = support.ub
        upper = support.lb
    end
    return RealInterval(shift + scale * lower, shift + scale * upper)
end


#### Conversions

function convert(::Type{AffineDistribution{T}}, shift::Real, scale::Real, base_dist::UnivariateDistribution) where T<:Real
    return AffineDistribution(T(shift), T(scale), base_dist)
end
function convert(::Type{AffineDistribution{T}}, d::AffineDistribution{S}) where {T<:Real, S<:Real} 
    # Should we really leave the type of d.base_dist unchanged? (Old LocationScale behavior)
    return affine(T(d.shift), T(d.scale), d.base_dist)
end

Base.eltype(::Type{<:AffineDistribution{T, S, D}}) where {T} = promote(T, eltype(D))


#### Parameters

function location(d::AffineDistribution)
    if hasmethod(location, (typeof(d.base_dist),))
        return d.shift + location(d.base_dist)
    else
        return d.shift
    end
end
function scale(d::AffineDistribution)  # This might be ambiguous... do we want to return signed or unsigned scale?
    if hasmethod(scale, (typeof(d.base_dist),))
        return d.scale * scale(d.base_dist)
    else
        return d.scale
    end
end
params(d::AffineDistribution) = (d.shift, d.scale, d.base_dist)
partype(::AffineDistribution{T}) where {T} = T


#### Statistics

mean(d::AffineDistribution) = d.shift + d.scale * mean(d.base_dist)
median(d::AffineDistribution) = d.shift + d.scale * median(d.base_dist)
mode(d::AffineDistribution) = d.shift + d.scale * mode(d.base_dist)
modes(d::AffineDistribution) = d.shift .+ d.scale .* modes(d.base_dist)

var(d::AffineDistribution) = d.scale^2 * var(d.base_dist)
std(d::AffineDistribution) = abs(d.scale) * std(d.base_dist)
skewness(d::AffineDistribution) = skewness(d.base_dist)
kurtosis(d::AffineDistribution) = kurtosis(d.base_dist)

isplatykurtic(d::AffineDistribution) = isplatykurtic(d.base_dist)
isleptokurtic(d::AffineDistribution) = isleptokurtic(d.base_dist)
ismesokurtic(d::AffineDistribution) = ismesokurtic(d.base_dist)

entropy(d::ContinuousAffine) = entropy(d.base_dist) + log(abs(d.scale))
entropy(d::DiscreteAffine) = entropy(d.base_dist)

mgf(d::AffineDistribution, t::Real) = exp(d.shift * t) * mgf(d.base_dist, d.scale * t)


#### Evaluation & Sampling

pdf(d::ContinuousAffine, x::Real) = pdf(d.base_dist, (x - d.shift) / d.scale) / abs(d.scale)
pdf(d::DiscreteAffine, x::Real) = pdf(d.base_dist, (x - d.shift) / d.scale)

function logpdf(d::ContinuousAffine, x::Real) 
    return logpdf(d.base_dist, (x - d.shift) / d.scale) - log(abs(d.scale))
end
logpdf(d::DiscreteAffine, x::Real) = logpdf(d.base_dist, (x - d.shift) / d.scale)

function cdf(d::AffineDistribution, x::Real)
    x = (x - d.shift) / d.scale
    return d.scale < 0 ? ccdf(d.base_dist, x) : cdf(d.base_dist, x)
end

function logcdf(d::AffineDistribution, x::Real)
    x = (x - d.shift) / d.scale
    return d.scale < 0 ? logccdf(d.base_dist, x) : logcdf(d.base_dist, x)
end

function quantile(d::AffineDistribution, q::Real)
    q = d.scale < 0 ? (1 - q) : q
    return d.shift + d.scale * quantile(d.base_dist, q)
end

rand(rng::AbstractRNG, d::AffineDistribution) = d.shift + d.scale * rand(rng, d.base_dist)
cf(d::AffineDistribution, t::Real) = cf(d.base_dist, t * d.scale) * exp(1im * t * d.shift)
gradlogpdf(d::ContinuousAffine, x::Real) = gradlogpdf(d.base_dist, (x - d.shift) / d.scale) / d.scale


#### Syntactic sugar for simple transforms of distributions, e.g., d + x, d - x, and so on

Base.:+(d::UnivariateDistribution, x::Real) = affine(x, one(x), d)
Base.:+(x::Real, d::UnivariateDistribution) = d + x
Base.:*(x::Real, d::UnivariateDistribution) = affine(zero(x), x, d)
Base.:*(d::UnivariateDistribution, x::Real) = x * d
Base.:-(d::UnivariateDistribution, x::Real) = d + -x
Base.:/(d::UnivariateDistribution, x::Real) = inv(x) * d