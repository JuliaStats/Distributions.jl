"""
    ReshapedDistribution(d::Distribution{<:ArrayLikeVariate}, dims::Dims{N})

Distribution of the `N`-dimensional random variable `reshape(X, dims)` where `X` is a random
variable with distribution `d`.

It is recommended to not use `reshape` instead of the constructor of `ReshapedDistribution`
directly since `reshape` can return more optimized distributions for specific types of `d`
and number of dimensions `N`.
"""
struct ReshapedDistribution{N,S<:ValueSupport,D<:Distribution{<:ArrayLikeVariate,S}} <: Distribution{ArrayLikeVariate{N},S}
    dist::D
    dims::Dims{N}

    function ReshapedDistribution(dist::Distribution{<:ArrayLikeVariate,S}, dims::Dims{N}) where {N,S<:ValueSupport}
        _reshape_check_dims(dist, dims)
        return new{N,S,typeof(dist)}(dist, dims)
    end
end

function _reshape_check_dims(dist::Distribution{<:ArrayLikeVariate}, dims::Dims)
    (all(d > 0 for d in dims) && length(dist) == prod(dims)) ||
        throw(ArgumentError("dimensions $(dims) do not match size of source distribution $(size(dist))"))
end

Base.size(d::ReshapedDistribution) = d.dims
Base.eltype(::Type{ReshapedDistribution{<:Any,<:ValueSupport,D}}) where {D} = eltype(D)

partype(d::ReshapedDistribution) = partype(d.dist)
params(d::ReshapedDistribution) = (d.dist, d.dims)

function insupport(d::ReshapedDistribution{N}, x::AbstractArray{<:Real,N}) where {N}
    return size(d) == size(x) && insupport(d.dist, reshape(x, size(d.dist)))
end

mean(d::ReshapedDistribution) = reshape(mean(d.dist), size(d))
var(d::ReshapedDistribution) = reshape(var(d.dist), size(d))
cov(d::ReshapedDistribution) = reshape(cov(d.dist), length(d), length(d))
function cov(d::ReshapedDistribution{2}, ::Val{false})
    n, p = size(d)
    return reshape(cov(d), n, p, n, p)
end

mode(d::ReshapedDistribution) = reshape(mode(d.dist), size(d))

# TODO: remove?
rank(d::ReshapedDistribution{2}) = minimum(size(d))

# logpdf evaluation
# have to fix method ambiguity due to default fallback for `MatrixDistribution`...
_logpdf(d::ReshapedDistribution{N}, x::AbstractArray{<:Real,N}) where {N} = __logpdf(d, x)
_logpdf(d::ReshapedDistribution{2}, x::AbstractMatrix{<:Real}) = __logpdf(d, x)
function __logpdf(d::ReshapedDistribution{N}, x::AbstractArray{<:Real,N}) where {N}
    dist = d.dist
    return @inbounds logpdf(dist, reshape(x, size(dist)))
end

# loglikelihood
# useful if the original distribution defined more optimized methods
@inline function loglikelihood(
    d::ReshapedDistribution{N},
    x::AbstractArray{<:Real,N},
) where {N}
    @boundscheck begin
        size(x) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    dist = d.dist
    return @inbounds loglikelihood(dist, reshape(x, size(dist)))
end
@inline function loglikelihood(
    d::ReshapedDistribution{N},
    x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of `x` ($M) must be greater than number of dimensions of `d` ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    dist = d.dist
    trailingsize = ntuple(i -> size(x, N + i), Val(M - N))
    return @inbounds loglikelihood(dist, reshape(x, size(dist)..., trailingsize...))
end

# sampling
function _rand!(
    rng::AbstractRNG,
    d::ReshapedDistribution{N},
    x::AbstractArray{<:Real,N}
) where {N}
    dist = d.dist
    @inbounds rand!(rng, dist, reshape(x, size(dist)))
    return x
end

"""
    reshape(d::Distribution{<:ArrayLikeVariate}, dims::Int...)
    reshape(d::Distribution{<:ArrayLikeVariate}, dims::Dims)

Return a [`Distribution`](@ref) of `reshape(X, dims)` where `X` is a random variable with
distribution `d`.

The default implementation returns a [`ReshapedDistribution`](@ref). However, it can return
more optimized distributions for specific types of distributions and numbers of dimensions.
Therefore it is recommended to use `reshape` instead of the constructor of
`ReshapedDistribution`.

# Implementation

Since `reshape(d, dims::Int...)` calls `reshape(d, dims::Dims)`, one should implement
`reshape(d, ::Dims)` for desired distributions `d`.

See also: [`vec`](@ref)
"""
function Base.reshape(dist::Distribution{<:ArrayLikeVariate}, dims::Dims)
    return ReshapedDistribution(dist, dims)
end
function Base.reshape(dist::Distribution{<:ArrayLikeVariate}, dims1::Int, dims::Int...)
    return reshape(dist, (dims1, dims...))
end

"""
    vec(d::Distribution{<:ArrayLikeVariate})

Return a [`MultivariateDistribution`](@ref) of `vec(X)` where `X` is a random variable with
distribution `d`.

The default implementation returns a [`ReshapedDistribution`](@ref). However, it can return
more optimized distributions for specific types of distributions and numbers of dimensions.
Therefore it is recommended to use `vec` instead of the constructor of
`ReshapedDistribution`.

# Implementation

Since `vec(d)` is defined as `reshape(d, length(d))` one should implement
`reshape(d, ::Tuple{Int})` rather than `vec`.

See also: [`reshape`](@ref)
"""
Base.vec(dist::Distribution{<:ArrayLikeVariate}) = reshape(dist, length(dist))

# avoid unnecessary wrappers
function Base.reshape(
    dist::ReshapedDistribution{<:Any,<:ValueSupport,<:MultivariateDistribution},
    dims::Tuple{Int},
)
    _reshape_check_dims(dist, dims)
    return dist.dist
end

function Base.reshape(dist::MultivariateDistribution, dims::Tuple{Int})
    _reshape_check_dims(dist, dims)
    return dist
end

# specialization for flattened `MatrixNormal`
function Base.reshape(dist::MatrixNormal, dims::Tuple{Int})
    _reshape_check_dims(dist, dims)
    return MvNormal(vec(dist.M), kron(dist.V, dist.U))
end
