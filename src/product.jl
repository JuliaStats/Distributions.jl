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
        return new{M + N, S, T, typeof(v)}(v)
    end
end

# aliases
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


## Vector of univariate distributions
length(d::VectorOfUnivariateDistribution) = length(d.v)

_rand!(rng::AbstractRNG, d::VectorOfUnivariateDistribution, x::AbstractVector{<:Real}) =
    broadcast!(dn->rand(rng, dn), x, d.v)
_logpdf(d::VectorOfUnivariateDistribution, x::AbstractVector{<:Real}) =
    sum(n->logpdf(d.v[n], x[n]), 1:length(d))

mean(d::VectorOfUnivariateDistribution) = map(mean, d.v)
var(d::VectorOfUnivariateDistribution) = map(var, d.v)
cov(d::VectorOfUnivariateDistribution) = Diagonal(var(d))
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
