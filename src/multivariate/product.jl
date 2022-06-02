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
    T<:Distribution{<:VariateForm,S},
    V<:AbstractVector{T},
} <: MultivariateDistribution{S}
    v::V
    function Product(v::V) where
        V<:AbstractVector{T} where
        T<:Distribution{<:VariateForm,S} where
        S<:ValueSupport
        return new{S, T, V}(v)
    end
end

length(d::Product) = sum(length.(d.v))
function Base.eltype(::Type{<:Product{S,T}}) where {S<:ValueSupport,
                                                    T<:Distribution{<:VariateForm,S}}
    # eltype(ContinuousDistribution) returns Any, necessitating this hack
    if T == ContinuousDistribution
        return Float64
    elseif T == DiscreteDistribution
        return Int
    else
        return eltype(T)
    end
end

function _rand!(rng::AbstractRNG, d::Product, x::AbstractVector{<:Real})
    if _isindependent(d)
        map!(Base.Fix1(rand, rng), x, d.v)
    else
        x = vcat(map(Base.Fix1(rand, rng), d.v)...)
        return x
    end
end

_logpdf(d::Product, x::AbstractVector{<:Real}) =
    sum(n->logpdf(d.v[n], _partitionargs(d, x)[n]), 1:length(d.v))

mean(d::Product) = _flatten(mean.(d.v))
var(d::Product) = _flatten(var.(d.v))

function cov(d::Product)
    if _isindependent(d)
        return Diagonal(var(d))
    else
        sparse_covs = map(d -> d isa UnivariateDistribution ? spdiagm(0 => [var(d)]) : sparse(cov(d)), d.v)
        return blockdiag(sparse_covs...)
    end
end

entropy(d::Product) = sum(entropy, d.v)
insupport(d::Product, x::AbstractVector) = all(insupport.(d.v, _partitionargs(d, x)))
minimum(d::Product) = _flatten(map(minimum, d.v))
maximum(d::Product) = _flatten(map(maximum, d.v))

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
    σ2 = var.(dists)
    return MvNormal(µ, Diagonal(σ2))
end

##### utility functions

_isindependent(d::Product) = all(length.(d.v) .== 1)

_flatten(v::AbstractVector) = all(length.(v) .== 1) ? v : vcat(v...)

# Split the vector x so that each constituent distribution can be evaluated
# correctly at the elements of the split vector
function _partitionargs(d::Product, x::AbstractVector{T}) where T<:Real
    dshape = length.(d.v)
    args = Vector{Union{T,Vector{T}}}(undef, length(dshape))

    start_idx = 1
    for (i, n) in enumerate(dshape)
        if n == 1
            args[i] = x[start_idx]
        else
            args[i] = collect(x[start_idx:(start_idx + n - 1)])
        end
        start_idx += n
    end
    return args
end
