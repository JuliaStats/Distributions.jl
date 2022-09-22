# Deprecated product distribution
# TODO: Remove in next breaking release

"""
    Product <: MultivariateDistribution

An N dimensional `MultivariateDistribution` constructed from a vector of N independent
`UnivariateDistribution`s.

```julia
Product(Uniform.(rand(10), 1)) # A 10-dimensional Product from 10 independent `Uniform` distributions.
```
"""
struct Product{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    V<:AbstractVector{T},
} <: MultivariateDistribution{S}
    v::V
    function Product{S,T,V}(v::V) where {S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractVector{T}}
        return new{S,T,V}(v)
    end
end

function Product(v::V) where {S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractVector{T}}
    Base.depwarn(
        "`Product(v)` is deprecated, please use `product_distribution(v)`",
        :Product,
    )
    return Product{S, T, V}(v)
end

length(d::Product) = length(d.v)
function Base.eltype(::Type{<:Product{S,T}}) where {S<:ValueSupport,
                                                    T<:UnivariateDistribution{S}}
    return eltype(T)
end

_rand!(rng::AbstractRNG, d::Product, x::AbstractVector{<:Real}) =
    map!(Base.Fix1(rand, rng), x, d.v)
function _logpdf(d::Product, x::AbstractVector{<:Real})
    dists = d.v
    if isempty(dists)
        return sum(map(logpdf, dists, x))
    end
    return sum(n -> logpdf(dists[n], x[n]), 1:length(d))
end

mean(d::Product) = mean.(d.v)
var(d::Product) = var.(d.v)
cov(d::Product) = Diagonal(var(d))
entropy(d::Product) = sum(entropy, d.v)
insupport(d::Product, x::AbstractVector) = all(insupport.(d.v, x))
minimum(d::Product) = map(minimum, d.v)
maximum(d::Product) = map(maximum, d.v)

# will be removed when `Product` is removed
# it will return a `ProductDistribution` then which is already the default for
# higher-dimensional arrays and distributions
function product_distribution(dists::V) where {S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractVector{T}}
    return Product{S,T,V}(dists)
end
