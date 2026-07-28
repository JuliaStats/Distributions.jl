"""
    ProductDistribution <: Distribution{<:ValueSupport,<:ArrayLikeVariate}

A distribution of `M + N`-dimensional arrays, constructed from an `N`-dimensional array of
independent `M`-dimensional distributions by stacking them.

Users should use [`product_distribution`](@ref) to construct a product distribution of
independent distributions instead of constructing a `ProductDistribution` directly.
"""
struct ProductDistribution{N,M,D,S<:ValueSupport} <: Distribution{ArrayLikeVariate{N},S}
    dists::D
    size::Dims{N}

    function ProductDistribution{N,M,D}(dists::D) where {N,M,D}
        if isempty(dists)
            throw(ArgumentError("a product distribution must consist of at least one distribution"))
        end
        return new{N,M,D,_product_valuesupport(dists)}(
            dists,
            _product_size(dists),
        )
    end
end

function ProductDistribution(dists::AbstractArray{<:Distribution{<:ArrayLikeVariate{M}},N}) where {M,N}
    return ProductDistribution{M + N,M,typeof(dists)}(dists)
end

function ProductDistribution(dists::Tuple{Distribution{<:ArrayLikeVariate{M}},Vararg{Distribution{<:ArrayLikeVariate{M}}}}) where {M}
    return ProductDistribution{M + 1,M,typeof(dists)}(dists)
end

# default definitions (type stable e.g. for arrays with concrete `eltype`)
_product_valuesupport(dists) = mapreduce(value_support ∘ typeof, promote_type, dists)

# type-stable and faster implementations for tuples
function _product_valuesupport(dists::NTuple{<:Any,Distribution})
    return __promote_type(value_support, typeof(dists))
end

_common_eltype(xs) = mapreduce(eltype, promote_type, xs)
_common_eltype(xs::NTuple{N,<:Real}) where {N} = __promote_type(eltype, typeof(xs))
__promote_type(f::F, ::Type{Tuple{D}}) where {F,D} = f(D)
function __promote_type(f::F, ::Type{T}) where {F,T<:Tuple}
    return promote_type(
        f(Base.tuple_type_head(T)),
        __promote_type(f, Base.tuple_type_tail(T)),
    )
end

function _product_size(dists::AbstractArray{<:Distribution{<:ArrayLikeVariate{M}},N}) where {M,N}
    size_d = size(first(dists))
    all(size(d) == size_d for d in dists) || error("all distributions must be of the same size")
    size_dists = size(dists)
    return ntuple(i -> i <= M ? size_d[i] : size_dists[i-M], Val(M + N))
end
function _product_size(dists::Tuple{Distribution{<:ArrayLikeVariate{M}},Vararg{Distribution{<:ArrayLikeVariate{M}}, N}}) where {M,N}
    size_d = size(first(dists))
    all(size(d) == size_d for d in dists) || error("all distributions must be of the same size")
    return ntuple(i -> i <= M ? size_d[i] : N + 1, Val(M + 1))
end

## aliases
const VectorOfUnivariateDistribution{D,S<:ValueSupport} = ProductDistribution{1,0,D,S}
const MatrixOfUnivariateDistribution{D,S<:ValueSupport} = ProductDistribution{2,0,D,S}
const ArrayOfUnivariateDistribution{N,D,S<:ValueSupport} = ProductDistribution{N,0,D,S}

const FillArrayOfUnivariateDistribution{N,D<:FillArrays.AbstractFill{<:Any,N},S<:ValueSupport} = ProductDistribution{N,0,D,S}

## General definitions
size(d::ProductDistribution) = d.size

mean(d::ProductDistribution) = reshape(mapreduce(vec ∘ mean, vcat, d.dists), size(d))
var(d::ProductDistribution) = reshape(mapreduce(vec ∘ var, vcat, d.dists), size(d))
cov(d::ProductDistribution) = Diagonal(vec(var(d)))

## For product distributions of univariate distributions
mean(d::ArrayOfUnivariateDistribution) = map(mean, d.dists)
mean(d::VectorOfUnivariateDistribution{<:Tuple}) = collect(map(mean, d.dists))
std(d::ArrayOfUnivariateDistribution) = map(std, d.dists)
std(d::VectorOfUnivariateDistribution{<:Tuple}) = collect(map(std, d.dists))
var(d::ArrayOfUnivariateDistribution) = map(var, d.dists)
var(d::VectorOfUnivariateDistribution{<:Tuple}) = collect(map(var, d.dists))

function insupport(d::ArrayOfUnivariateDistribution{N}, x::AbstractArray{<:Real,N}) where {N}
    size(d) == size(x) && all(insupport(vi, xi) for (vi, xi) in zip(d.dists, x))
end

minimum(d::ArrayOfUnivariateDistribution) = map(minimum, d.dists)
minimum(d::VectorOfUnivariateDistribution{<:Tuple}) = collect(map(minimum, d.dists))
maximum(d::ArrayOfUnivariateDistribution) = map(maximum, d.dists)
maximum(d::VectorOfUnivariateDistribution{<:Tuple}) = collect(map(maximum, d.dists))

function entropy(d::ArrayOfUnivariateDistribution)
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    return sum(Broadcast.instantiate(Broadcast.broadcasted(entropy, d.dists)))
end
# fix type instability with tuples
entropy(d::VectorOfUnivariateDistribution{<:Tuple}) = sum(entropy, d.dists)

## Vector of univariate distributions
length(d::VectorOfUnivariateDistribution) = length(d.dists)

## For matrix distributions
cov(d::ProductDistribution{2}, ::Val{false}) = reshape(cov(d), size(d)..., size(d)...)

# Arrays of univariate distributions
function rand(rng::AbstractRNG, d::ArrayOfUnivariateDistribution)
    x = map(Base.Fix1(rand, rng), d.dists)
    if x isa AbstractArray{<:Real}
        return x
    else
        # For instance, if x is a tuple
        return collect(_common_eltype(x), x)
    end
end
@inline function rand!(
    rng::AbstractRNG,
    d::ArrayOfUnivariateDistribution{N},
    x::AbstractArray{<:Real,N},
) where {N}
    @boundscheck size(x) == size(d)
    map!(Base.Fix1(rand, rng), x, d.dists)
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
    # without allocations to compute `sum(logpdf.(d.dists, x))`
    broadcasted = Broadcast.broadcasted(logpdf, d.dists, x)
    return sum(Broadcast.instantiate(broadcasted))
end

# more efficient sampling for `Fill` array of univariate distributions
function rand(
    rng::AbstractRNG,
    d::FillArrayOfUnivariateDistribution,
)
    return rand(rng, first(d.dists), size(d))
end
@inline function rand!(
    rng::AbstractRNG,
    d::FillArrayOfUnivariateDistribution{N},
    x::AbstractArray{<:Real,N},
) where {N}
    @boundscheck size(x) == size(d)
    rand!(rng, first(d.dists), x)
    return x
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
    return loglikelihood(first(d.dists), x)
end

# sampling for arrays of distributions
function rand(rng::AbstractRNG, d::ProductDistribution)
    x = mapreduce(vec ∘ Base.Fix1(rand, rng), hcat, d.dists)
    return reshape(x, size(d))
end
@inline function rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,M},
    A::AbstractArray{<:Real,N},
) where {N,M}
    @boundscheck size(A) == size(d)
    for (di, Ai) in zip(d.dists, eachvariate(A, ArrayLikeVariate{M}))
        rand!(rng, di, Ai)
    end
    return A
end

# `_logpdf` for arrays of distributions
# we have to fix some method ambiguities
_logpdf(d::ProductDistribution{N}, x::AbstractArray{<:Real,N}) where {N} = __logpdf(d, x)
_logpdf(d::ProductDistribution{2}, x::AbstractMatrix{<:Real}) = __logpdf(d, x)
function __logpdf(
    d::ProductDistribution{N,M},
    x::AbstractArray{<:Real,N},
) where {N,M}
    # we use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    # to compute `sum(logpdf.(d.dists, eachvariate))`
    broadcasted = Broadcast.broadcasted(
        logpdf, d.dists, eachvariate(x, ArrayLikeVariate{M}),
    )
    return sum(Broadcast.instantiate(broadcasted))
end

# more efficient sampling for `Fill` arrays of distributions
function rand(rng::AbstractRNG, d::ProductDistribution{<:Any,<:Any,<:FillArrays.AbstractFill})
    return _product_rand(rng, sampler(first(d.dists)), size(d))
end
function _product_rand(rng::AbstractRNG, spl::Sampleable{ArrayLikeVariate{N}}, dims::Dims) where N
    xi = rand(rng, spl)
    x = Array{eltype(xi)}(undef, dims)
    copyto!(x, xi)
    vx = reshape(x, ntuple(i -> i <= N ? size(xi, i) : Colon(), N + 1))
    @inbounds rand!(rng, spl, @view(vx[ntuple(i -> i <= N ? Colon() : 2:lastindex(vx, N + 1), N + 1)...]))
    return x
end

@inline function rand!(
    rng::AbstractRNG,
    d::ProductDistribution{N,M,<:FillArrays.AbstractFill},
    A::AbstractArray{<:Real,N},
) where {N,M}
    @boundscheck size(A) == size(d)
    rand!(rng, sampler(first(d.dists)), A)
    return A
end

# more efficient implementation of `_logpdf` for `AbstractFill` arrays of distributions
# we have to fix a method ambiguity
function _logpdf(
    d::ProductDistribution{N,M,<:FillArrays.AbstractFill},
    x::AbstractArray{<:Real,N},
) where {N,M}
    return __logpdf(d, x)
end
function _logpdf(
    d::ProductDistribution{2,M,<:FillArrays.AbstractFill},
    x::AbstractMatrix{<:Real},
) where {M}
    return __logpdf(d, x)
end
function __logpdf(
    d::ProductDistribution{N,M,<:FillArrays.AbstractFill},
    x::AbstractArray{<:Real,N},
) where {N,M}
    return loglikelihood(first(d.dists), x)
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

function product_distribution(
    dist::Distribution{ArrayLikeVariate{N}}, dists::Distribution{ArrayLikeVariate{N}}...,
) where {N}
    return ProductDistribution((dist, dists...))
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
