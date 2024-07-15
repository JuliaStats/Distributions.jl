"""
    ProductNamedTupleDistribution{Tnames,Tdists,S<:ValueSupport,eltypes} <:
        Distribution{NamedTupleVariate{Tnames},S}

    A distribution of `NamedTuple`s, constructed from a `NamedTuple` of independent named
    distributions.

    Users should use [`product_distribution`](@ref) to construct a product distribution of
    independent distributions instead of constructing a `ProductNamedTupleDistribution`
    directly.
"""
struct ProductNamedTupleDistribution{Tnames,Tdists,S<:ValueSupport,eltypes} <:
       Distribution{NamedTupleVariate{Tnames},S}
    dists::NamedTuple{Tnames,Tdists}
end
function ProductNamedTupleDistribution(
    dists::NamedTuple{K,V}
) where {K,V<:Tuple{Distribution,Vararg{Distribution}}}
    vs = _product_valuesupport(values(dists))
    eltypes = _product_namedtuple_eltype(values(dists))
    return ProductNamedTupleDistribution{K,V,vs,eltypes}(dists)
end

_gentype(d::UnivariateDistribution) = eltype(d)
_gentype(d::Distribution{<:ArrayLikeVariate{S}}) where {S} = Array{eltype(d),S}
function _gentype(d::Distribution{CholeskyVariate})
    T = eltype(d)
    return LinearAlgebra.Cholesky{T,Matrix{T}}
end
_gentype(::Distribution) = Any

_product_namedtuple_eltype(dists) = typejoin(map(_gentype, dists)...)

function Base.show(io::IO, d::ProductNamedTupleDistribution)
    return show_multline(io, d, collect(pairs(d.dists)))
end

function distrname(::ProductNamedTupleDistribution{K}) where {K}
    return "ProductNamedTupleDistribution{$K}"
end

"""
    product_distribution(dists::Namedtuple{K,Tuple{Vararg{Distribution}}}) where {K}

Create a distribution of `NamedTuple`s as a product distribution of independent named
distributions.

The function falls back to constructing a [`ProductNamedTupleDistribution`](@ref)
distribution but specialized methods can be defined.
"""
function product_distribution(
    dists::NamedTuple{<:Any,<:Tuple{Distribution,Vararg{Distribution}}}
)
    return ProductNamedTupleDistribution(dists)
end

# Properties

Base.eltype(::Type{<:ProductNamedTupleDistribution{<:Any,<:Any,<:Any,T}}) where {T} = T

Base.minimum(d::ProductNamedTupleDistribution) = map(minimum, d.dists)

Base.maximum(d::ProductNamedTupleDistribution) = map(maximum, d.dists)

function insupport(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return all(map(insupport, dist.dists, x))
end

# Evaluation

function pdf(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return exp(logpdf(dist, x))
end

function logpdf(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return sum(map(logpdf, dist.dists, x))
end

function loglikelihood(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return logpdf(dist, x)
end

function loglikelihood(
    dist::ProductNamedTupleDistribution{K}, xs::AbstractArray{<:NamedTuple{K}}
) where {K}
    return sum(Base.Fix1(loglikelihood, dist), xs)
end

# Statistics

mode(d::ProductNamedTupleDistribution) = map(mode, d.dists)

mean(d::ProductNamedTupleDistribution) = map(mean, d.dists)

var(d::ProductNamedTupleDistribution) = map(var, d.dists)

std(d::ProductNamedTupleDistribution) = map(std, d.dists)

entropy(d::ProductNamedTupleDistribution) = sum(entropy, values(d.dists))

function kldivergence(
    d1::ProductNamedTupleDistribution{K}, d2::ProductNamedTupleDistribution{K}
) where {K}
    return mapreduce(kldivergence, +, d1.dists, d2.dists)
end

# Sampling

function sampler(d::ProductNamedTupleDistribution{K,<:Any,S}) where {K,S}
    samplers = map(sampler, d.dists)
    Tsamplers = typeof(values(samplers))
    return ProductNamedTupleSampler{K,Tsamplers,S}(samplers)
end

function Base.rand(rng::AbstractRNG, d::ProductNamedTupleDistribution{K}) where {K}
    return NamedTuple{K}(map(Base.Fix1(rand, rng), d.dists))
end
function Base.rand(
    rng::AbstractRNG, d::ProductNamedTupleDistribution{K}, dims::Dims
) where {K}
    return convert(AbstractArray{<:NamedTuple{K}}, _rand(rng, sampler(d), dims))
end

function _rand!(rng::AbstractRNG, d::ProductNamedTupleDistribution, xs::AbstractArray)
    return _rand!(rng, sampler(d), xs)
end
