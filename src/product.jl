import Statistics: mean, var, cov

"""
    ProductDistribution <: Distribution{<:ValueSupport,<:ArrayLikeVariate}

A distribution of `M + N`-dimensional arrays, constructed from an `N`-dimensional array of
independent `M`-dimensional distributions by stacking them.

Users should use [`product_distribution`](@ref) to construct a product distribution of
independent distributions instead of constructing a `ProductDistribution` directly.
"""
struct ProductDistribution{
    N,
    S<:ValueSupport,
    T<:Distribution{<:ArrayLikeVariate,S},
    V<:AbstractArray{T},
} <: Distribution{ArrayLikeVariate{N},S}
    v::V

    function ProductDistribution(v::AbstractArray{T,N}) where {S<:ValueSupport, M, T<:Distribution{ArrayLikeVariate{M},S}, N}
        isempty(v) &&
            error("product distribution must consist of at least one distribution")
        return new{M + N, S, T, typeof(v)}(v)
    end
end

## aliases
const VectorOfUnivariateDistribution{S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractVector{T}} =
    ProductDistribution{1,S,T,V}
const MatrixOfUnivariateDistribution{S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractMatrix{T}} =
    ProductDistribution{2,S,T,V}
const VectorOfMultivariateDistribution{S<:ValueSupport,T<:MultivariateDistribution{S},V<:AbstractVector{T}} =
    ProductDistribution{2,S,T,V}

## deprecations
# type parameters can't be deprecated it seems: https://github.com/JuliaLang/julia/issues/9830
# so we define an alias and deprecate the corresponding constructor
const Product{S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractVector{T}} = ProductDistribution{1,S,T,V}
Base.@deprecate Product(v::AbstractVector{<:UnivariateDistribution}) ProductDistribution(v)

## General definitions
function Base.eltype(::Type{<:ProductDistribution{N,S,T}}) where {N,S<:ValueSupport,T<:Distribution{<:ArrayLikeVariate,S}}
    return eltype(T)
end

function size(d::ProductDistribution{N}) where {N}
    size_d = size(first(d.v))
    size_v = size(d.v)
    M = length(size_d)
    return ntuple(i -> i <= M ? size_d[i] : size_v[i - M], Val(N))
end

mean(d::ProductDistribution) = reshape(mapreduce(vec ∘ mean, vcat, d.v), size(d))
var(d::ProductDistribution) = reshape(mapreduce(vec ∘ var, vcat, d.v), size(d))
cov(d::ProductDistribution) = Diagonal(vec(var(d)))

## For product distributions of univariate distributions
function mean(d::ProductDistribution{N,S,<:UnivariateDistribution{S}}) where {N,S<:ValueSupport}
    return map(mean, d.v)
end
function var(d::ProductDistribution{N,S,<:UnivariateDistribution{S}}) where {N,S<:ValueSupport}
    return map(var, d.v)
end

function _rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,S,<:UnivariateDistribution{S}},
    x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport}
    @inbounds for (i, di) in zip(eachindex(x), d.v)
        x[i] = rand(rng, di)
    end
    return x
end
function _logpdf(
    d::ProductDistribution{N,S,T,<:AbstractArray{T,N}}, x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport,T<:UnivariateDistribution{S}}
    return sum(logpdf(di, xi) for (di, xi) in zip(d.v, x))
end

# possibly more efficient implementations for `Fill` arrays
function _rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,S,T,<:Fill{T,N}},
    x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport,T<:UnivariateDistribution{S}}
    return rand!(rng, sampler(first(d.v)), x)
end
function _logpdf(
    d::ProductDistribution{N,S,T,<:Fill{T,N}}, x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport,T<:UnivariateDistribution{S}}
    return loglikelihood(first(d.v), x)
end

## Product distributions of multivariate distributions
function _rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,S,<:MultivariateDistribution{S}},
    A::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport}
    B = view(A, size(A, 1), :)
    @inbounds for (i, di) in zip(axes(B, 2), d.v)
        rand!(rng, di, view(B, :, i))
    end
    return A
end
function _logpdf(
    d::ProductDistribution{N,S,<:MultivariateDistribution{S}},
    x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport}
    y = view(x, size(x, 1), :)
    return sum(logpdf(di, yi) for (di, yi) in zip(d.v, (view(y, :, i) for i in axes(y, 2))))
end

# possibly more efficient implementations for `Fill` arrays
function _rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,S,T,<:Fill{T}},
    A::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport,T<:MultivariateDistribution{S}}
    di = first(d.v)
    rand!(rng, sampler(di), reshape(A, length(di), :))
    return A
end
function _logpdf(
    d::ProductDistribution{N,S,T,<:Fill{T}},
    x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport,T<:MultivariateDistribution{S}}
    di = first(d.v)
    loglikelihood(di, reshape(x, length(di), :))
end

## For matrix distributions
cov(d::ProductDistribution{2}, ::Val{false}) = reshape(cov(d), size(d)..., size(d)...)

## Vector of univariate distributions
length(d::VectorOfUnivariateDistribution) = length(d.v)


entropy(d::VectorOfUnivariateDistribution) = sum(entropy, d.v)
function insupport(d::VectorOfUnivariateDistribution, x::AbstractVector)
    length(d) == length(x) && all(insupport(vi, xi) for (vi, xi) in zip(d.v, x))
end

"""
    product_distribution(dists::AbstractArray{<:Distribution{<:ArrayLikeVariate{M}},N})

Create a distribution of `M + N`-dimensional arrays as a product distribution of
independent `M`-dimensional distributions by stacking them.

The function falls back to constructing a [`ProductDistribution`](@ref) distribution but
specialized methods can be defined.
"""
function product_distribution(dists::AbstractArray{<:Distribution{<:ArrayLikeVariate}})
    return ProductDistribution(dists)
end

"""
    product_distribution(dists::AbstractVector{<:Normal})

Create a multivariate normal distribution by stacking the univariate normal distributions.

The resulting distribution of type [`MvNormal`](@ref) has a diagonal covariance matrix.
"""
function product_distribution(dists::AbstractVector{<:Normal})
    µ = map(mean, dists)
    σ2 = map(var, dists)
    return MvNormal(µ, Diagonal(σ2))
end
