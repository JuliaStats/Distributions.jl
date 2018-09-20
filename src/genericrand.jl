### Generic rand methods

"""
    rand(s::Sampleable)

Generate one sample for `s`.

    rand(s::Sampleable, n::Int)

Generate `n` samples from `s`. The form of the returned object depends on the variate form of `s`:

- When `s` is univariate, it returns a vector of length `n`.
- When `s` is multivariate, it returns a matrix with `n` columns.
- When `s` is matrix-variate, it returns an array, where each element is a sample matrix.
"""
rand(s::Sampleable)

"""
    rand!(s::Sampleable, A::AbstractArray)

Generate one or multiple samples from `s` to a pre-allocated array `A`. `A` should be in the
form as specified above. The rules are summarized as below:

- When `s` is univariate, `A` can be an array of arbitrary shape. Each element of `A` will
  be overriden by one sample.
- When `s` is multivariate, `A` can be a vector to store one sample, or a matrix with each
  column for a sample.
- When `s` is matrix-variate, `A` can be a matrix to store one sample, or an array of
  matrices with each element for a sample matrix.
"""
rand!(s::Sampleable, A::AbstractArray)

# univariate

function _rand!(s::Sampleable{Univariate}, A::AbstractArray)
    for i in 1:length(A)
        @inbounds A[i] = rand(s)
    end
    return A
end
rand!(s::Sampleable{Univariate}, A::AbstractArray) = _rand!(s, A)

rand(s::Sampleable{Univariate}, dims::Dims) =
    _rand!(s, Array{eltype(s)}(undef, dims))

rand(s::Sampleable{Univariate}, dims::Int...) =
    _rand!(s, Array{eltype(s)}(undef, dims))


# multivariate

function _rand!(s::Sampleable{Multivariate}, A::AbstractMatrix)
    for i = 1:size(A,2)
        _rand!(s, view(A,:,i))
    end
    return A
end

function rand!(s::Sampleable{Multivariate}, A::AbstractVector)
    length(A) == length(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand!(s, A)
end

function rand!(s::Sampleable{Multivariate}, A::AbstractMatrix)
    size(A,1) == length(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand!(s, A)
end

rand(s::Sampleable{Multivariate}) =
    _rand!(s, Vector{eltype(s)}(undef, length(s)))

rand(s::Sampleable{Multivariate}, n::Int) =
    _rand!(s, Matrix{eltype(s)}(undef, length(s), n))


# matrix-variate

function _rand!(s::Sampleable{Matrixvariate}, X::AbstractArray{M}) where M<:Matrix
    for i in 1:length(X)
        X[i] = rand(s)
    end
    return X
end

rand!(s::Sampleable{Matrixvariate}, X::AbstractArray{M}) where {M<:Matrix} =
    _rand!(s, X)

rand(s::Sampleable{Matrixvariate}, n::Int) =
    rand!(s, Vector{Matrix{eltype(s)}}(n))

"""
    sampler(d::Distribution) -> Sampleable

Samplers can often rely on pre-computed quantities (that are not parameters themselves) to improve efficiency.
If such a sampler exists, it can be provide with this `sampler` method, which would be used for batch sampling.
The general fallback is `sampler(d::Distribution) = d`.
"""
sampler(d::Distribution) = d
