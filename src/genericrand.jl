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

# deprecated fallbacks (univariate distributions have their own fallback)
function rand(rng::AbstractRNG, s::Sampleable{<:ArrayLikeVariate})
    _depwarn_rand(s)
    return rand!(rng, s, Array{eltype(s)}(undef, size(s)))
end
# workaround for sampleables that incorrectly base `eltype` on the parameters
function rand(rng::AbstractRNG, s::Sampleable{<:ArrayLikeVariate,Continuous})
    _depwarn_rand(s)
    return rand!(rng, s, Array{float(eltype(s))}(undef, size(s)))
end
function _depwarn_rand(s::Sampleable)
    Base.depwarn(
        "The default fallback for `rand(rng::AbstractRNG, s::$(typeof(s)))` is deprecated and will be removed. Please implement it.",
        :rand,
    )
    return nothing
end

# multiple samples
# we use function barriers since for some distributions `sampler(s)` is not type-stable:
# https://github.com/JuliaStats/Distributions.jl/pull/1281
function rand(rng::AbstractRNG, s::Sampleable, dims::Dims)
    return _rand(rng, sampler(s), dims)
end
# The first sample determines the element type of the output.
function _rand(rng::AbstractRNG, s::Sampleable, dims::Dims)
    r = rand(rng, s)
    out = Array{typeof(r)}(undef, dims)
    isempty(out) && return out
    out[1] = r
    rand!(rng, s, @view(out[2:end]))
    return out
end
# For univariate samplers the first sample is discarded: filling the whole `Array` is
# considerably faster than filling a view since some methods in Random (e.g. `randn!`
# and `randexp!`) are only optimized for `Array`s.
function _rand(rng::AbstractRNG, s::Sampleable{Univariate}, dims::Dims)
    out = Array{typeof(rand(rng, s))}(undef, dims)
    return rand!(rng, s, out)
end

# Throw a `DimensionMismatch` unless `x` can store one sample (`M == N`) or multiple
# samples along its trailing dimensions (`M > N`) of `s`
function _check_rand_size(s::Sampleable{ArrayLikeVariate{N}}, x::AbstractArray{<:Any,M}) where {N,M}
    if M < N || ntuple(i -> size(x, i), Val(N)) != size(s)
        throw(DimensionMismatch(lazy"samples of size $(size(s)) cannot be stored in an array of size $(size(x))"))
    end
    return nothing
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
rand!(s::Sampleable{<:ArrayLikeVariate}, x::AbstractArray{<:Real}) = rand!(default_rng(), s, x)
function rand!(s::Sampleable{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}}) where {N}
    return rand!(default_rng(), s, x)
end
function rand!(
    s::Sampleable{CholeskyVariate},
    x::Union{LinearAlgebra.Cholesky,AbstractArray{<:LinearAlgebra.Cholesky}},
)
    return rand!(default_rng(), s, x)
end
rand!(s::Sampleable{<:NamedTupleVariate}, x::AbstractArray{<:NamedTuple}) = rand!(default_rng(), s, x)

# default definitions for arraylike variates
function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,N},
) where {N}
    Base.depwarn(
        "The default fallback for `rand!(rng::AbstractRNG, s::$(typeof(s)), x::AbstractArray{<:Real,$N})` is deprecated and will be removed. Please implement it.",
        :rand!,
    )
    _check_rand_size(s, x)
    return _rand!(rng, s, x)
end

function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,M},
) where {N,M}
    _check_rand_size(s, x)
    # the function barrier fixes performance issues if `sampler(s)` is type unstable
    _rand!(rng, sampler(s), x)
    return x
end

function _rand!(
    rng::AbstractRNG,
    s::Sampleable{<:ArrayLikeVariate},
    x::AbstractArray{<:Real},
)
    for xi in eachvariate(x, variate_form(typeof(s)))
        rand!(rng, s, xi)
    end
    return x
end

function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    sz = size(s)
    allocate = !all(isassigned(x, i) && size(x[i]) == sz for i in eachindex(x))
    return rand!(rng, s, x, allocate)
end

function rand!(
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
    allocate::Bool,
) where {N}
    return rand!(default_rng(), s, x, allocate)
end
function rand!(
    rng::AbstractRNG,
    s::Sampleable{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N}},
    allocate::Bool,
) where {N}
    if !allocate
        foreach(Base.Fix1(_check_rand_size, s), x)
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
        for i in eachindex(x)
            x[i] = rand(rng, s)
        end
    else
        for xi in x
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
