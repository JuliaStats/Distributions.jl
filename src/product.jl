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
const ArrayOfUnivariateDistribution{N,S<:ValueSupport,T<:UnivariateDistribution{S},V<:AbstractArray{T,N}} =
    ProductDistribution{N,S,T,V}
const FillArrayOfUnivariateDistribution{N,S<:ValueSupport,T<:UnivariateDistribution{S},V<:Fill{T,N}} =
    ProductDistribution{N,S,T,V}
const VectorOfMultivariateDistribution{S<:ValueSupport,T<:MultivariateDistribution{S},V<:AbstractVector{T}} =
    ProductDistribution{2,S,T,V}
const ArrayOfMultivariateDistribution{N,S<:ValueSupport,T<:MultivariateDistribution{S},V<:AbstractArray{T}} =
    ProductDistribution{N,S,T,V}
const FillArrayOfMultivariateDistribution{N,S<:ValueSupport,T<:MultivariateDistribution{S},V<:Fill{T}} =
    ProductDistribution{N,S,T,V}


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
mean(d::ArrayOfUnivariateDistribution) = map(mean, d.v)
var(d::ArrayOfUnivariateDistribution) = map(var, d.v)
function insupport(d::ArrayOfUnivariateDistribution{N}, x::AbstractArray{<:Real,N}) where {N}
    size(d) == size(x) && all(insupport(vi, xi) for (vi, xi) in zip(d.v, x))
end

minimum(d::ArrayOfUnivariateDistribution) = map(minimum, d.v)
maximum(d::ArrayOfUnivariateDistribution) = map(maximum, d.v)

function entropy(d::ArrayOfUnivariateDistribution)
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    return sum(Broadcast.instantiate(Broadcast.broadcasted(entropy, d.v)))
end

## Vector of univariate distributions
length(d::VectorOfUnivariateDistribution) = length(d.v)

## For matrix distributions
cov(d::ProductDistribution{2}, ::Val{false}) = reshape(cov(d), size(d)..., size(d)...)

# `_rand!` for arrays of univariate distributions
function _rand!(
    rng::AbstractRNG,
    d::ArrayOfUnivariateDistribution{N},
    x::AbstractArray{<:Real,N},
) where {N}
    @inbounds for (i, di) in zip(eachindex(x), d.v)
        x[i] = rand(rng, di)
    end
    return x
end

# `_logpdf` for arrays of univariate distributions
# we have to fix a method ambiguity
function _logpdf(d::ArrayOfUnivariateDistribution, x::AbstractArray{<:Real,N}) where {N}
    return __logpdf(d, x)
end
_logpdf(d::MatrixOfUnivariateDistribution, x::AbstractMatrix{<:Real}) = __logpdf(d, x)
function __logpdf(d::ArrayOfUnivariateDistribution, x::AbstractArray{<:Real,N}) where {N}
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    # without allocations to compute `sum(logpdf.(d.v, x))`
    broadcasted = Broadcast.broadcasted(logpdf, d.v, x)
    return sum(Broadcast.instantiate(broadcasted))
end

# more efficient implementation of `_rand!` for `Fill` array of univariate distributions
function _rand!(
    rng::AbstractRNG,
    d::FillArrayOfUnivariateDistribution{N},
    x::AbstractArray{<:Real,N},
) where {N}
    return @inbounds rand!(rng, sampler(first(d.v)), x)
end

# more efficient implementation of `_logpdf` for `Fill` array of univariate distributions
# we have to fix a method ambiguity
function _logpdf(
    d::FillArrayOfUnivariateDistribution{N}, x::AbstractArray{<:Real,N}
) where {N}
    return __logpdf(d, x)
end
_logpdf(d::FillArrayOfUnivariateDistribution{2}, x::AbstractMatrix{<:Real}) = __logpdf(d, x)
function __logpdf(
    d::FillArrayOfUnivariateDistribution{N}, x::AbstractArray{<:Real,N}
) where {N}
    return @inbounds loglikelihood(first(d.v), x)
end

# `_rand! for arrays of distributions
function _rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,S,<:Distribution{ArrayLikeVariate{M},S}},
    A::AbstractArray{<:Real,N},
) where {N,M,S<:ValueSupport}
    @inbounds for (di, Ai) in zip(d.v, eachvariate(A, ArrayLikeVariate{M}))
        rand!(rng, di, Ai)
    end
    return A
end

# `_logpdf` for arrays of distributions
# we have to fix a method ambiguity
_logpdf(d::ProductDistribution{N}, x::AbstractArray{<:Real,N}) where {N} = __logpdf(d, x)
_logpdf(d::ProductDistribution{2}, x::AbstractMatrix{<:Real}) = __logpdf(d, x)
function __logpdf(
    d::ProductDistribution{N,S,<:Distribution{ArrayLikeVariate{M},S}},
    x::AbstractArray{<:Real,N},
) where {N,M,S<:ValueSupport}
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    # to compute `sum(logpdf.(d.v, eachvariate))`
    @inbounds broadcasted = Broadcast.broadcasted(
        logpdf, d.v, eachvariate(x, ArrayLikeVariate{M}),
    )
    return sum(Broadcast.instantiate(broadcasted))
end

# more efficient implementation of `_rand!` for `Fill` arrays of distributions
function _rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,S,T,<:Fill{T}},
    A::AbstractArray{<:Real,N},
) where {N,M,S<:ValueSupport,T<:Distribution{ArrayLikeVariate{M},S}}
    @inbounds rand!(rng, sampler(first(d.v)), A)
    return A
end

# more efficient implementation of `_logpdf` for `Fill` arrays of distributions
# we have to fix a method ambiguity
function _logpdf(
    d::ProductDistribution{N,S,T,<:Fill{T}},
    x::AbstractArray{<:Real,N},
) where {N,S<:ValueSupport,T<:Distribution{<:ArrayLikeVariate,S}}
    return __logpdf(d, x)
end
function _logpdf(
    d::ProductDistribution{2,S,T,<:Fill{T}},
    x::AbstractMatrix{<:Real},
) where {S<:ValueSupport,T<:Distribution{<:ArrayLikeVariate,S}}
    return __logpdf(d, x)
end
function __logpdf(
    d::ProductDistribution{N,S,T,<:Fill{T}},
    x::AbstractArray{<:Real,N},
) where {N,M,S<:ValueSupport,T<:Distribution{ArrayLikeVariate{M},S}}
    return @inbounds loglikelihood(first(d.v), x)
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
