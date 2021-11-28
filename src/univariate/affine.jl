@deprecate LocationScale(args...; kwargs...) affine(args...)

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
Otherwise, it will fall back on the `Affine` wrapper type.

```julia
affine(shift, scale, base_dist)  # location-scale transformed distribution
params(d)            # Get the parameters, i.e. (shift, scale, base_dist)
location(d)          # Get the location parameter
scale(d)             # Get the scale parameter
```
"""
struct Affine{T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}} <: UnivariateDistribution{S}
    shift::T
    scale::T
    base_dist::D
    function Affine{T,S,D}(shift::T, scale::T, base_dist::D; check_args=true) where {
        T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}
    }
        !check_args || @check_args(Affine, !iszero(scale))
        return new{T,S,D}(shift, scale, base_dist)
    end
end

function affine(shift::Real, scale::Real, base_dist::UnivariateDistribution)
    T = promote_type(eltype(base_dist), typeof(shift), typeof(scale))
    D = typeof(base_dist)
    S = value_support(D)
    return Affine{T,S,D}(T(shift), T(scale), base_dist)
end

# Composing affine transformations
function affine(shift::Real, scale::Real, d::Affine)
    return affine(shift + scale * d.shift, scale * d.scale, d.base_dist)
end


# Aliases
const ContinuousAffine{T<:Real,D<:ContinuousUnivariateDistribution} = Affine{T,Continuous,D}
const DiscreteAffine{T<:Real,D<:DiscreteUnivariateDistribution} = Affine{T,Discrete,D}

Base.eltype(::Type{<:Affine{T}}) where {T} = T

# alias μ, σ, ρ for backwards compatibility with old LocationScale
function Base.getproperty(x::Affine, name::Symbol)
    try
        getfield(x, name)
    catch
        if name == :μ
            return x.shift
        elseif name == :σ
            return x.scale
        elseif name == :ρ
            return x.base_dist
        else
            error("type Affine has no field $name")
        end
    end
end

# Support
function minimum(d::Affine)
    minim = d.scale > 0 ? minimum : maximum
    return d.shift + d.scale * minim(d.base_dist)
end
function maximum(d::Affine)
    maxim = d.scale > 0 ? maximum : minimum
    return d.shift + d.scale * maxim(d.base_dist)
end
support(d::Affine) = affine_support(d.shift, d.scale, support(d.base_dist))

affine_support(shift::Real, scale::Real, support) = shift .+ scale .* support
function affine_support(shift::Real, scale::Real, support::RealInterval)
    if scale > 0
        lower = support.lb
        upper = support.ub
    else
        lower = support.ub
        upper = support.lb
    end
    _transform = x -> shift + scale * x
    return RealInterval(_transform(lower), _transform(upper))
end


#### Conversions

function convert(::Type{Affine{T}}, shift::Real, scale::Real, base_dist::UnivariateDistribution) where T<:Real
    return Affine(T(shift), T(scale), base_dist)
end
function convert(::Type{Affine{T}}, d::Affine{S}) where {T<:Real, S<:Real} 
    # Should we really leave the type of d.base_dist unchanged? (Old LocationScale behavior)
    return affine(T(d.shift), T(d.scale), d.base_dist)
end

#### Parameters

function location(d::Affine)
    if hasmethod(location, (typeof(d.base_dist),))
        return d.shift + location(d.base_dist)
    else
        return d.shift
    end
end
function scale(d::Affine)  # This might be ambiguous... do we want to return signed or unsigned scale?
    if hasmethod(scale, (typeof(d.base_dist),))
        return d.scale * scale(d.base_dist)
    else
        return d.scale
    end
end
params(d::Affine) = (d.shift, d.scale, d.base_dist)
partype(::Affine{T}) where {T} = T

#### Statistics

mean(d::Affine) = d.shift + d.scale * mean(d.base_dist)
median(d::Affine) = d.shift + d.scale * median(d.base_dist)
mode(d::Affine) = d.shift + d.scale * mode(d.base_dist)
modes(d::Affine) = d.shift .+ d.scale .* modes(d.base_dist)

var(d::Affine) = d.scale^2 * var(d.base_dist)
std(d::Affine) = abs(d.scale) * std(d.base_dist)
skewness(d::Affine) = skewness(d.base_dist)
kurtosis(d::Affine) = kurtosis(d.base_dist)

isplatykurtic(d::Affine) = isplatykurtic(d.base_dist)
isleptokurtic(d::Affine) = isleptokurtic(d.base_dist)
ismesokurtic(d::Affine) = ismesokurtic(d.base_dist)

entropy(d::ContinuousAffine) = entropy(d.base_dist) + log(abs(d.scale))
entropy(d::DiscreteAffine) = entropy(d.base_dist)

mgf(d::Affine, t::Real) = exp(d.shift * t) * mgf(d.base_dist, d.scale * t)

#### Evaluation & Sampling

pdf(d::ContinuousAffine, x::Real) = pdf(d.base_dist, (x - d.shift) / d.scale) / abs(d.scale)
pdf(d::DiscreteAffine, x::Real) = pdf(d.base_dist, (x - d.shift) / d.scale)

function logpdf(d::ContinuousAffine, x::Real) 
    return logpdf(d.base_dist, (x - d.shift) / d.scale) - log(abs(d.scale))
end
logpdf(d::DiscreteAffine, x::Real) = logpdf(d.base_dist, (x - d.shift) / d.scale)

function cdf(d::Affine, x::Real)
    x = (x - d.shift) / d.scale
    return d.scale < 0 ? ccdf(d.base_dist, x) : cdf(d.base_dist, x)
end

function logcdf(d::Affine, x::Real)
    x = (x - d.shift) / d.scale
    return d.scale < 0 ? logccdf(d.base_dist, x) : logcdf(d.base_dist, x)
end

function quantile(d::Affine, q::Real)
    q = d.scale < 0 ? (1 - q) : q
    return d.shift + d.scale * quantile(d.base_dist, q)
end

rand(rng::AbstractRNG, d::Affine) = d.shift + d.scale * rand(rng, d.base_dist)
cf(d::Affine, t::Real) = cf(d.base_dist, t * d.scale) * exp(1im * t * d.shift)
gradlogpdf(d::ContinuousAffine, x::Real) = gradlogpdf(d.base_dist, (x - d.shift) / d.scale) / d.scale

#### Syntactic sugar for simple transforms of distributions, e.g., d + x, d - x, and so on

Base.:+(d::UnivariateDistribution, x::Real) = affine(x, one(x), d)
Base.:+(x::Real, d::UnivariateDistribution) = d + x
Base.:*(x::Real, d::UnivariateDistribution) = affine(zero(x), x, d)
Base.:*(d::UnivariateDistribution, x::Real) = x * d
Base.:-(d::UnivariateDistribution, x::Real) = d + -x
Base.:/(d::UnivariateDistribution, x::Real) = inv(x) * d
