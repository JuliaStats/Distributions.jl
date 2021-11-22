### Generic rand methods

"""
    rand(rng::AbstractRNG=GLOBAL_RNG, ::Type{T}=eltype(s), s::Sampleable)

Generate one sample for `s` of elment type `T`.
"""
rand(s::Sampleable) = rand(eltype(s), s)
rand(::Type{T}, s::Sampleable) where {T} = rand(GLOBAL_RNG, T, s)

"""
    rand(rng::AbstractRNG=GLOBAL_RNG, ::Type{T}=eltype(s), s::Sampleable, n::Int)

Generate `n` samples from `s` of element type `T`.

The form of the returned object depends on the variate form of `s`:
- When `s` is univariate, it returns a vector of length `n`.
- When `s` is multivariate, it returns a matrix with `n` columns.
- When `s` is matrix-variate, it returns an array, where each element is a sample matrix.
"""
rand(s::Sampleable, n::Int) = rand(eltype(s), s, n)
rand(::Type{T}, s::Sampleable, n::Int) where {T} = rand(GLOBAL_RNG, T, s, n)
rand(rng::AbstractRNG, ::Type{T}, s::Sampleable, n::Int) where {T} = rand(rng, T, s, (n,))

"""
    rand(rng::AbstractRNG=GLOBAL_RNG, ::Type{T}=eltype(s), s::Sampleable, dims::Int...)
    rand(rng::AbstractRNG=GLOBAL_RNG, ::Type{T}=eltype(s), s::Sampleable, dims::Dims)

Generate an array of samples from `s` of element type `T` whose shape is determined by the given
dimensions.
"""
rand(s::Sampleable, dims::Dims) = rand(eltype(s), s, dims)
function rand(s::Sampleable, dims1::Int, dims2::Int, dims::Int...)
    return rand(eltype(s), s, dims1, dims2, dims...)
end
function rand(::Type{T}, s::Sampleable, dims1::Int, dims2::Int, dims::Int...) where {T}
    return rand(T, s, (dims1, dims2, dims...))
end
rand(::Type{T}, s::Sampleable, dims::Dims) where {T} = rand(GLOBAL_RNG, T, s, dims)

"""
    rand!([rng::AbstractRNG,] s::Sampleable, A::AbstractArray)

Generate one or multiple samples from `s` to a pre-allocated array `A`. `A` should be in the
form as specified above. The rules are summarized as below:

- When `s` is univariate, `A` can be an array of arbitrary shape. Each element of `A` will
  be overriden by one sample.
- When `s` is multivariate, `A` can be a vector to store one sample, or a matrix with each
  column for a sample.
- When `s` is matrix-variate, `A` can be a matrix to store one sample, or an array of
  matrices with each element for a sample matrix.
"""
function rand! end
rand!(s::Sampleable, X::AbstractArray{<:AbstractArray}, allocate::Bool) =
    rand!(GLOBAL_RNG, s, X, allocate)
rand!(s::Sampleable, X::AbstractArray) = rand!(GLOBAL_RNG, s, X)
rand!(rng::AbstractRNG, s::Sampleable, X::AbstractArray) = _rand!(rng, s, X)

"""
    sampler(s::Sampleable)

Return a sampler that is used for batch sampling.

Samplers can often rely on pre-computed quantities (that are not parameters
themselves) to improve efficiency. The general fallback is `sampler(s) = s`.
"""
sampler(s::Sampleable) = s

# Random API
Random.Sampler(::Type{<:AbstractRNG}, s::Sampleable, ::Val{1}) = s
Random.Sampler(::Type{<:AbstractRNG}, s::Sampleable, ::Val{Inf}) = sampler(s)
