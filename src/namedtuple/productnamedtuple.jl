struct ProductNamedTupleDistribution{Tnames,Tdists,eltypes,S<:ValueSupport} <:
       Distribution{NamedTupleVariate{Tnames},S}
    dists::NamedTuple{Tnames,Tdists}
end
function ProductNamedTupleDistribution(
    dists::NamedTuple{K,V}
) where {K,V<:Tuple{Vararg{Distribution}}}
    eltypes = Tuple{map(eltype, values(dists))...}
    # TODO: allow mixed ValueSupports here
    vs = _product_valuesupport(dists)
    return ProductNamedTupleDistribution{K,V,eltypes,vs}(dists)
end

function Base.show(io::IO, d::ProductNamedTupleDistribution)
    show_multline(io, d, collect(pairs(d.dists)))
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
function product_distribution(dists::NamedTuple{<:Any,<:Tuple{Vararg{Distribution}}})
    return ProductNamedTupleDistribution(dists)
end

# Properties

function Base.eltype(::Type{<:ProductNamedTupleDistribution{K,<:Any,V}}) where {K,V}
    return NamedTuple{K,V}
end

function minimum(
    d::ProductNamedTupleDistribution{<:Any,<:Tuple{Vararg{UnivariateDistribution}}}
)
    return map(minimum, d.dists)
end
function maximum(
    d::ProductNamedTupleDistribution{<:Any,<:Tuple{Vararg{UnivariateDistribution}}}
)
    return map(maximum, d.dists)
end

function insupport(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return all(Base.splat(insupport), zip(dist.dists, x))
end

# Evaluation

function pdf(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return exp(logpdf(dist, x))
end
function logpdf(dist::ProductNamedTupleDistribution{K}, x::NamedTuple{K}) where {K}
    return mapreduce(logpdf, +, dist.dists, x)
end

# Statistics

mode(d::ProductNamedTupleDistribution) = map(mode, d.dists)

mean(d::ProductNamedTupleDistribution) = map(mean, d.dists)

var(d::ProductNamedTupleDistribution) = map(var, d.dists)

entropy(d::ProductNamedTupleDistribution) = sum(entropy, d.dists)

function kldivergence(
    d1::ProductNamedTupleDistribution{K}, d2::ProductNamedTupleDistribution{K}
) where {K}
    return mapreduce(kldivergence, +, d1.dists, d2.dists)
end

# Sampling

function Base.rand(rng::AbstractRNG, d::ProductNamedTupleDistribution{K}) where {K}
    return NamedTuple{K}(map(Base.Fix1(rand, rng), d.dists))
end
function Base.rand(rng::AbstractRNG, d::ProductNamedTupleDistribution, dims::Dims)
    x = rand(rng, d)
    xs = Array{typeof(x)}(undef, dims)
    xs[1] = x
    for i in Iterators.drop(eachindex(xs), 1)
        xs[i] = rand(rng, d)
    end
    return xs
end

function _rand!(
    rng::AbstractRNG,
    d::ProductNamedTupleDistribution,
    xs::AbstractArray,
)
    for i in eachindex(xs)
        xs[i] = Random.rand(rng, d)
    end
    return xs
end
