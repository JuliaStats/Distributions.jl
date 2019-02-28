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
rand(s::Sampleable) = rand(GLOBAL_RNG, s)
rand(s::Sampleable, dims::Dims) = rand(GLOBAL_RNG, s, dims)
rand(s::Sampleable, dim1::Int, moredims::Int...) =
    rand(GLOBAL_RNG, s, (dim1, moredims...))
rand(rng::AbstractRNG, s::Sampleable, dim1::Int, moredims::Int...) =
    rand(rng, s, (dim1, moredims...))

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
    sampler(d::Distribution) -> Sampleable
    sampler(s::Sampleable) -> s

Samplers can often rely on pre-computed quantities (that are not parameters
themselves) to improve efficiency. If such a sampler exists, it can be provided
with this `sampler` method, which would be used for batch sampling.
The general fallback is `sampler(d::Distribution) = d`.
"""
sampler(s::Sampleable) = s
