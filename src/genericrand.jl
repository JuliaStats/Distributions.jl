### Generic rand methods

"""
    rand([rng::AbstractRNG,] s::Sampleable)

Generate one sample for `s`.

    rand([rng::AbstractRNG,] s::Sampleable, n::Int)

Generate `n` samples from `s`. The form of the returned object depends on the variate form of `s`:

- When `s` is univariate, it returns a vector of length `n`.
- When `s` is multivariate, it returns a matrix with `n` columns.
- When `s` is matrix-variate, it returns an array, where each element is a sample matrix.

    rand([rng::AbstractRNG,] s::Sampleable, dim1::Int, dim2::Int...)
    rand([rng::AbstractRNG,] s::Sampleable, dims::Dims)

Generate an array of samples from `s` whose shape is determined by the given
dimensions.
"""
rand(s::Sampleable, dims::Int...) = rand(default_rng(), s, dims...)
rand(s::Sampleable, dims::Dims) = rand(default_rng(), s, dims)
rand(rng::AbstractRNG, s::Sampleable, dim1::Int, moredims::Int...) =
    rand(rng, s, (dim1, moredims...))

# default fallback (redefined for univariate distributions)
function rand(rng::AbstractRNG, s::Sampleable{<:ArrayLikeVariate})
    return @inbounds rand!(rng, s, Array{eltype(s)}(undef, size(s)))
end

# multiple samples
function rand(rng::AbstractRNG, s::Sampleable{Univariate}, dims::Dims)
    out = Array{eltype(s)}(undef, dims)
    return @inbounds rand!(rng, sampler(s), out)
end
function rand(
    rng::AbstractRNG, s::Sampleable{<:ArrayLikeVariate}, dims::Dims,
)
    sz = size(s)
    ax = map(Base.OneTo, dims)
    out = [Array{eltype(s)}(undef, sz) for _ in Iterators.product(ax...)]
    return @inbounds rand!(rng, sampler(s), out, false)
end

# these are workarounds for sampleables that incorrectly base `eltype` on the parameters
function rand(rng::AbstractRNG, s::Sampleable{<:ArrayLikeVariate,Continuous})
    return @inbounds rand!(rng, sampler(s), Array{float(eltype(s))}(undef, size(s)))
end
function rand(rng::AbstractRNG, s::Sampleable{Univariate,Continuous}, dims::Dims)
    out = Array{float(eltype(s))}(undef, dims)
    return @inbounds rand!(rng, sampler(s), out)
end
function rand(
    rng::AbstractRNG, s::Sampleable{<:ArrayLikeVariate,Continuous}, dims::Dims,
)
    sz = size(s)
    ax = map(Base.OneTo, dims)
    out = [Array{float(eltype(s))}(undef, sz) for _ in Iterators.product(ax...)]
    return @inbounds rand!(rng, sampler(s), out, false)
end

"""
    rand!([rng::AbstractRNG,] s::Sampleable, A::AbstractArray)

Generate one or multiple samples from `s` to a pre-allocated array `A`. `A` should be in the
form as specified above. The rules are summarized as below:

- When `s` is univariate, `A` can be an array of arbitrary shape. Each element of `A` will
  be overridden by one sample.
- When `s` is multivariate, `A` can be a vector to store one sample, or a matrix with each
  column for a sample.
- When `s` is matrix-variate, `A` can be a matrix to store one sample, or an array of
  matrices with each element for a sample matrix.
"""
function rand! end
Base.@propagate_inbounds rand!(s::Sampleable, X::AbstractArray) = rand!(default_rng(), s, X)
Base.@propagate_inbounds function rand!(rng::AbstractRNG, s::Sampleable, X::AbstractArray)
    return _rand!(rng, s, X)
end

# default definitions for arraylike variates
@inline function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,N},
) where {N}
    @boundscheck begin
        size(x) == size(s) || throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _rand!(rng, s, x)
end

@inline function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of `x` ($M) must be greater than number of dimensions of `s` ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(s) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    # the function barrier fixes performance issues if `sampler(s)` is type unstable
    return _rand!(rng, sampler(s), x)
end

function _rand!(
    rng::AbstractRNG,
    s::Sampleable{<:ArrayLikeVariate},
    x::AbstractArray{<:Real},
)
    @inbounds for xi in eachvariate(x, variate_form(typeof(s)))
        rand!(rng, s, xi)
    end
    return x
end

Base.@propagate_inbounds function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    sz = size(s)
    allocate = !all(isassigned(x, i) && size(@inbounds x[i]) == sz for i in eachindex(x))
    return rand!(rng, s, x, allocate)
end

Base.@propagate_inbounds function rand!(
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
    allocate::Bool,
) where {N}
    return rand!(default_rng(), s, x, allocate)
end
@inline function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
    allocate::Bool,
) where {N}
    @boundscheck begin
        if !allocate
            sz = size(s)
            all(size(xi) == sz for xi in x) ||
                throw(DimensionMismatch("inconsistent array dimensions"))
        end
    end
    # the function barrier fixes performance issues if `sampler(s)` is type unstable
    return _rand!(rng, sampler(s), x, allocate)
end

function _rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
    allocate::Bool,
) where {N}
    if allocate
        @inbounds for i in eachindex(x)
            x[i] = rand(rng, s)
        end
    else
        @inbounds for xi in x
            rand!(rng, s, xi)
        end
    end
    return x
end

"""
    sampler(d::Distribution) -> Sampleable
    sampler(s::Sampleable) -> s

Samplers can often rely on pre-computed quantities (that are not parameters
themselves) to improve efficiency. If such a sampler exists, it can be provided
with this `sampler` method, which would be used for batch sampling.
The general fallback is `sampler(d::Distribution) = d`.
"""
sampler(s::Sampleable) = s

# Random API
Random.Sampler(::Type{<:AbstractRNG}, s::Sampleable, ::Val{1}) = s
Random.Sampler(::Type{<:AbstractRNG}, s::Sampleable, ::Val{Inf}) = sampler(s)
