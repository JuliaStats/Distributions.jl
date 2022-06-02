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
                                                    T<:UnivariateDistribution{S}}
    return eltype(T)
end

function _rand!(rng::AbstractRNG, d::Product, x::AbstractVector{<:Real})
    if all(length.(d.v) .== 1)
        map!(Base.Fix1(rand, rng), x, d.v)
    else
        dshape = length.(d.v)
        start_idx = 1
        for (i, n) in enumerate(dshape)
            if n == 1
                x[start_idx - 1] = rand(rng, d.v[i])
            else
                x[start_idx:(start_idx + n - 1)] = rand(rng, d.v[i])
            end
            start_idx += n
        end
        return x
    end
end

_logpdf(d::Product, x::AbstractVector{<:Real}) =
    sum(n->logpdf(d.v[n], _partitionargs(d, x)[n]), 1:length(d.v))

mean(d::Product) = _flatten(mean.(d.v))
var(d::Product) = _flatten(var.(d.v))

function cov(d::Product)
    if all(length.(d.v) .== 1)
        return Diagonal(var(d))
    else        
        sparse_covs = Vector{SparseMatrixCSC{Float64,Int64}}(undef, length(d.v))
        for (i, u) in enumerate(d.v)
            if typeof(u)<:UnivariateDistribution
                sparse_covs[i] = spdiagm(0 => [var(u)])
            elseif typeof(u)<:MultivariateDistribution
                sparse_covs[i] = sparse(cov(u))
            end
        end
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

# Supplementary functions

_flatten(v::AbstractVector) = all(length.(v) .== 1) ? v : vcat(v...)

function _partitionargs(d::Product, x::AbstractVector{T}) where T<:Real
    dshape = length.(d.v)
    args = Vector{Union{T,Vector{T}}}(undef, length(dshape))

    start_idx = 1
    for (i, n) in enumerate(dshape)
        if n == 1
            args[i] = x[start_idx]
        else
            args[i] = x[start_idx:(start_idx + n - 1)]
        end
        start_idx += n
    end
    return args
end
