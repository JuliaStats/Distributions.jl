"""
    AffineDistribution(μ, σ, ρ)

A shifted and scaled (affinely transformed) version of `ρ`.

If `Z` is a random variable with distribution `ρ`, then `AffineDistribution(μ, σ, ρ)` is
the distribution of the random variable
```math
X = μ + σ * Z
```

If `ρ` is a discrete univariate distribution, the probability mass function of the 
transformed distribution is given by
```math
P(X = x) = P\\left(Z = \\frac{x-μ}{σ} \\right).
```

If `ρ` is a continuous univariate distribution with probability density function `f_Z`,
the probability density function of the transformed distribution is given by
```math
f_X(x) = \\frac{1}{|σ|} f_Z\\left( \\frac{x-μ}{σ} \\right).
```

We recommend against using the `AffineDistribution` constructor directly. Instead, use 
`+`, `-`, `*`, and `/`. These are optimized for specific distributions and will automatically
fall back on `AffineDistribution` if they need to.

Affine transformations of discrete variables are easily affected by rounding errors. If you
are getting incorrect results, try using exact `Rational` types instead of floats.

```julia
d = σ * ρ + μ       # Create location-scale transformed distribution
params(d)           # Get the parameters, i.e. (μ, σ, ρ)
```
"""
struct AffineDistribution{
    F<:ArrayLikeVariate,
    S<:ValueSupport,
    Tμ<:Union{Real, AbstractArray},
    Tσ<:Union{Real, AbstractArray},
    D<:Distribution{F,S}
} <: Distribution{F,S} 
    μ::Tμ
    σ::Tσ
    ρ::D
end

function AffineDistribution(μ, σ, ρ; nonnegative=false)
    if nonnegative && σ ≤ 0
        throw(ArgumentError("scale cannot be negative"))
    end
    return AffineDistribution(μ, σ, ρ)
end


#### Aliases
const LocationScale{T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}} = 
    AffineDistribution{Univariate, S, T, T, D}
@deprecate LocationScale(args...; check_args=true) AffineDistribution(args...; nonnegative=!check_args)
const ContinuousAffine{F,Tμ,Tσ,D} = AffineDistribution{F,Continuous,Tμ,Tσ,D}
const DiscreteAffine{F,Tμ,Tσ,D} = AffineDistribution{F,Discrete,Tμ,Tσ,D}

const UnivariateAffine{S,Tμ,Tσ,D} = AffineDistribution{Univariate,S,Tμ,Tσ,D}
const MultivariateAffine{S,Tμ,Tσ,D} = AffineDistribution{Multivariate,S,Tμ,Tσ,D}


#### Constructors

"""
    +(μ, ρ)

Return a version of `ρ` that has been translated by `μ`.
"""
Base.:+(μ::Real, ρ::UnivariateDistribution) = AffineDistribution(μ, one(μ), ρ)

function Base.:+(μ::AbstractArray{<:Real, N}, ρ::Distribution{<:ArrayLikeVariate{N}}) where N
    if size(μ) ≠ size(ρ) 
        throw(DimensionMismatch("array and distribution have different sizes."))
    end
    return AffineDistribution(μ, one(eltype(μ)), ρ)
end


"""
    *(σ, ρ)

Return a version of `ρ` that has been scaled by `σ`, where `σ` can be any real number or 
matrix.
"""
function Base.:*(σ::Real, ρ::UnivariateDistribution)
    if iszero(σ)
        throw(ArgumentError("scale must be non-zero"))
    end
    return AffineDistribution(zero(σ), σ, ρ)
end

function Base.:*(σ::Real, ρ::Distribution{<:ArrayLikeVariate})
    if iszero(σ)
        throw(ArgumentError("scale must be non-zero"))
    end
    return AffineDistribution(Zeros{eltype(σ)}(size(ρ)), σ, ρ)
end

function Base.:*(σ::AbstractMatrix, ρ::MultivariateDistribution)
    if iszero(σ)
        throw(ArgumentError("scale must be non-zero"))
    end
    return AffineDistribution(Zeros{eltype(σ)}(σ), σ, ρ)
end


# Composing affine distributions

Base.:+(μ::Real, d::UnivariateAffine) = AffineDistribution(μ + d.μ, d.σ, d.ρ)
function Base.:+(μ::AbstractArray{<:Real, N}, d::AffineDistribution{F}) where {
        N, F<:ArrayLikeVariate{N}
    }
    return AffineDistribution(μ + d.μ, d.σ, d.ρ)
end


Base.:*(σ::Real, d::UnivariateAffine) = AffineDistribution(σ * d.μ, σ * d.σ, d.ρ)
Base.:*(σ::Real, d::MultivariateAffine) = AffineDistribution(σ * d.μ, σ * d.σ, d.ρ)
function Base.:*(σ::AbstractMatrix, d::MultivariateAffine)
    return AffineDistribution(σ * d.μ, σ * d.σ, d.ρ)
end


Base.:+(d::Distribution{<:ArrayLikeVariate}, μ::Union{Real, AbstractArray{<:Real}}) = μ + d
Base.:-(d::Distribution{<:ArrayLikeVariate}) = -1 * d
Base.:-(d::Distribution{<:ArrayLikeVariate}, μ::Union{Real, AbstractArray{<:Real}}) = -μ + d
Base.:-(μ::Union{Real, AbstractArray{<:Real}}, d::Distribution{<:ArrayLikeVariate}) = μ + -d

Base.:*(d::Distribution{<:ArrayLikeVariate}, σ::Real) = σ * d
Base.:/(d::Distribution{<:ArrayLikeVariate}, τ::Real) = inv(τ) * d
Base.:\(τ::Real, d::Distribution{<:ArrayLikeVariate}) = inv(τ) * d
Base.:\(τ::AbstractMatrix, d::MultivariateDistribution) = inv(τ) * d



#### Extremes

## Univariate

function maximum(d::AffineDistribution)
    maxim = d.σ > 0 ? maximum(d.ρ) : minimum(d.ρ)
    return d.μ + d.σ * maxim
end

function minimum(d::AffineDistribution)
    minim = d.σ > 0 ? minimum(d.ρ) : maximum(d.ρ)
    return d.μ + d.σ * minim
end

function extrema(d::UnivariateAffine)
    μ, σ, ρ = params(d)
    min_ρ, max_ρ = extrema(ρ)
    return minmax(μ + σ * min_ρ, μ + σ * max_ρ)
end


## Multivariate
# when finding the extrema, we need to flip the extremes if the scale is negative
_flip(signed::Real, tuple::Tuple) = signed > 0 ? tuple : (last(tuple), first(tuple))

function minimum(d::MultivariateAffine)
    extremes = extrema(d.ρ) |> zip |> collect |> permutedims
    # The minimum of d.ρ[i] will minimize d[j] if d.σ[i,j] is negative. 
    # Otherwise the maximum will minimize d[j]
    extreme_matrix = _flip.(d.σ, extremes)
    argmins = first.(extreme_matrix)
    mins = [σ ⋅ argmin for (σ, argmin) in zip(eachrow(d.σ), eachrow(argmins))]
    @. mins += d.μ
    return mins
end

function maximum(d::MultivariateAffine)
    extremes = extrema(d.ρ) |> zip |> collect |> permutedims
    extreme_matrix = _flip.(d.σ, extremes)
    argmaxes = last.(extreme_matrix)
    maxes = [σ ⋅ argmax for (σ, argmax) in zip(eachrow(d.σ), eachrow(argmaxes))]
    @. maxes += d.μ
    return maxes
end

function extrema(d::MultivariateAffine)
    extremes = extrema(d.ρ) |> zip |> collect |> permutedims
    extreme_matrix = _flip.(d.σ, extremes)
    argmins = first.(extreme_matrix)
    mins = [σ ⋅ argmin for (σ, argmin) in zip(eachrow(d.σ), eachrow(argmins))]
    @. mins += d.μ
    argmaxes = last.(extreme_matrix)
    maxes = [σ ⋅ argmax for (σ, argmax) in zip(eachrow(d.σ), eachrow(argmaxes))]
    @. maxes += d.μ
    return (mins, maxes)
end

support(d::AffineDistribution) = affine_support(d.μ, d.σ, support(d.ρ))
function affine_support(μ::Real, σ::Real, support)
    support = σ < 0 ? reverse(support) : copy(support)
    return μ .+ σ .* support
end


function affine_support(μ::Real, σ::Real, support::RealInterval) 
    lower, upper = extrema(support)
    return RealInterval(minmax(μ + σ * lower, μ + σ * upper)...)
end


#### Conversions

function convert(::Type{AffineDistribution{F,S,Tμ,Tσ,D}}, d::AffineDistribution) where {
        F<:VariateForm,
        S<:ValueSupport,
        D<:Distribution{F, S},
        Tμ<:Union{Real, AbstractArray{<:Real}},
        Tσ<:Union{Real, AbstractArray{<:Real}}
    }
    return AffineDistribution(Tμ(d.μ), Tσ(d.σ), convert(D, d.ρ))
end


function Base.eltype(::Type{<:AffineDistribution{F,S,Tμ,Tσ,D}}) where {
        F<:VariateForm,
        S<:ValueSupport,
        D<:Distribution{F, S},
        Tμ<:Union{Real, AbstractArray{<:Real}},
        Tσ<:Union{Real, AbstractArray{<:Real}}
    }
    T = Core.Compiler.return_type(Base.:*, (Tσ, eltype(D)))
    return Core.Compiler.return_type(Base.:+, (Tμ, T))
end


#### Parameters

function location(d::AffineDistribution)
    if hasmethod(location, (typeof(d.ρ),))
        return d.μ + d.σ * location(d.ρ)
    else
        return d.μ
    end
end
function scale(d::AffineDistribution)
    if hasmethod(scale, (typeof(d.ρ),))
        return d.σ * scale(d.ρ)
    else
        return d.σ
    end
end
shape(d::AffineDistribution) = shape(d.ρ)
ncategories(d::AffineDistribution) = ncategories(d.ρ)
params(d::AffineDistribution) = (d.μ, d.σ, d.ρ)
partype(::AffineDistribution{F,S,Tμ,Tσ,D}) where {F,S,Tμ,Tσ,D} = promote_type(Tμ, Tσ)


#### Statistics

mean(d::AffineDistribution) = d.μ + d.σ * mean(d.ρ)
median(d::AffineDistribution) = d.μ + d.σ * median(d.ρ)
mode(d::AffineDistribution) = d.μ + d.σ * mode(d.ρ)
modes(d::AffineDistribution) = map(x -> d.μ + d.σ * x, modes(d.ρ))

var(d::UnivariateAffine) = d.σ^2 * var(d.ρ)
std(d::UnivariateAffine) = abs(d.σ) * std(d.ρ)
skewness(d::UnivariateAffine) = sign(d.σ) * skewness(d.ρ)
kurtosis(d::UnivariateAffine) = kurtosis(d.ρ)

cov(d::MultivariateAffine{S,Tμ,Tσ,D}) where {S,Tμ,Tσ<:Diagonal,D} = d.σ^2 * cov(d.ρ)
cov(d::MultivariateAffine) = d.σ * cov(d.ρ) * d.σ'

isplatykurtic(d::AffineDistribution) = isplatykurtic(d.ρ)
isleptokurtic(d::AffineDistribution) = isleptokurtic(d.ρ)
ismesokurtic(d::AffineDistribution) = ismesokurtic(d.ρ)

log_abs_det(x::AbstractMatrix) = first(logabsdet(x))
log_abs_det(x::Real) = log(abs(x))

entropy(d::ContinuousAffine) = entropy(d.ρ) + log_abs_det(d.σ)
entropy(d::DiscreteAffine) = entropy(d.ρ)

mgf(d::AffineDistribution, t::Real) = exp(d.μ * t) * mgf(d.ρ, d.σ * t)
cf(d::AffineDistribution, t::Real) = cf(d.ρ, t * d.σ) * exp(1im * t * d.μ)


#### Evaluation & Sampling

# Continuous
function pdf(d::ContinuousAffine{Univariate}, x::Real) 
    return pdf(d.ρ, d.σ \ (x - d.μ)) / abs(d.σ)
end
function pdf(d::ContinuousAffine{<:ArrayLikeVariate{N}}, x::AbstractArray{N}) where N
    return pdf(d.ρ, d.σ \ (x - d.μ)) / abs(det(d.σ))
end

function logpdf(d::ContinuousAffine{Univariate}, x::Real) 
    return logpdf(d.ρ, d.σ \ (x - d.μ)) - log_abs_det(d.σ)
end
function logpdf(d::ContinuousAffine{<:ArrayLikeVariate{N}}, x::AbstractArray{N}) where N
    return logpdf(d.ρ, d.σ \ (x - d.μ)) - log_abs_det(d.σ)
end

# Discrete

function pdf(d::DiscreteAffine{Univariate}, x::Real)
    return pdf(d.ρ, d.σ \ (x - d.μ))
end
function pdf(d::DiscreteAffine{<:ArrayLikeVariate{N}}, x::AbstractArray{N}) where N
    return pdf(d.ρ, d.σ \ (x - d.μ))
end
function logpdf(d::DiscreteAffine{Univariate}, x::Real)
    return logpdf(d.ρ, d.σ \ (x - d.μ))
end
function logpdf(d::DiscreteAffine{<:ArrayLikeVariate{N}}, x::AbstractArray{N}) where N
    return logpdf(d.ρ, d.σ \ (x - d.μ))
end

# CDF methods

function cdf(d::UnivariateAffine, x::Real)
    x = (x - d.μ) / d.σ
    return d.σ < 0 ? ccdf(d.ρ, x) : cdf(d.ρ, x)
end

function logcdf(d::UnivariateAffine, x::Real)
    x = (x - d.μ) / d.σ
    return d.σ < 0 ? logccdf(d.ρ, x) : logcdf(d.ρ, x)
end 

function quantile(d::UnivariateAffine, q::Real)
    q = d.σ < 0 ? (1 - q) : q
    return d.μ + d.σ * quantile(d.ρ, q)
end

rand(rng::AbstractRNG, d::AffineDistribution) = d.μ + d.σ * rand(rng, d.ρ)
function rand!(rng::AbstractRNG, d::AffineDistribution, storage::AbstractArray)
    rand!(rng, d.ρ, storage)
    return @. storage = d.μ + d.σ * storage
end

function gradlogpdf(d::ContinuousAffine, x::Real)
    return d.σ \ gradlogpdf(d.ρ, d.σ \ (x - d.μ))
end
