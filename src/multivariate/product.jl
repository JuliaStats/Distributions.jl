import Statistics: mean, var, cov

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
    function Product(v::V) where
        V<:AbstractVector{T} where
        T<:UnivariateDistribution{S} where
        S<:ValueSupport
        return new{S, T, V}(v)
    end
end

length(d::Product) = length(d.v)
function Base.eltype(::Type{<:Product{S,T}}) where {S<:ValueSupport,
                                                    T<:UnivariateDistribution{S}}
    return eltype(T)
end

_rand!(rng::AbstractRNG, d::Product, x::AbstractVector{<:Real}) =
    broadcast!(dn->rand(rng, dn), x, d.v)
_logpdf(d::Product, x::AbstractVector{<:Real}) =
    sum(n->logpdf(d.v[n], x[n]), 1:length(d))

mean(d::Product) = mean.(d.v)
var(d::Product) = var.(d.v)
cov(d::Product) = Diagonal(var(d))
entropy(d::Product) = sum(entropy, d.v)
insupport(d::Product, x::AbstractVector) = all(insupport.(d.v, x))
minimum(d::Product) = map(minimum, d.v)
maximum(d::Product) = map(maximum, d.v)

"""
    product_distribution(dists::AbstractVector{<:UnivariateDistribution})

Creates a multivariate product distribution `P` from a vector of univariate distributions.
Fallback is the `Product constructor`, but specialized methods can be defined
for distributions with a special multivariate product.
"""
function product_distribution(dists::AbstractVector{<:UnivariateDistribution})
    return Product(dists)
end

"""
    product_distribution(dists::AbstractVector{<:Normal})

Computes the multivariate Normal distribution obtained by stacking the univariate
normal distributions. The result is a multivariate Gaussian with a diagonal
covariance matrix.
"""
function product_distribution(dists::AbstractVector{<:Normal})
    µ = mean.(dists)
    σ = std.(dists)
    return MvNormal(µ, σ)
end
