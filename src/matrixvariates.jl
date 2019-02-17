
##### Generic methods #####

"""
    size(d::MatrixDistribution)

Return the size of each sample from distribution `d`.
"""
size(d::MatrixDistribution)

"""
    length(d::MatrixDistribution)

The length (*i.e* number of elements) of each sample from the distribution `d`.
"""
Base.length(d::MatrixDistribution)

"""
    mean(d::MatrixDistribution)

Return the mean matrix of `d`.
"""
mean(d::MatrixDistribution)

## sampling
# multiple matrix-variates with pre-allocated array
function rand(rng::AbstractRNG, s::Sampleable{Matrixvariate},
              X::AbstractArray{M}) where M <: AbstractMatrix
    smp = sampler(s)
    for i in eachindex(X)
        X[i] = _rand!(rng, smp, M(undef, size(s)))
    end
    return X
end

# multiple matrix-variates with pre-allocated array of pre-allocated matrices
function rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
               X::AbstractArray{<:AbstractMatrix})
    smp = sampler(s)
    for x in X
        rand!(rng, smp, x)
    end
    return X
end

# multiple matrix-variates, must allocate array of arrays
rand(rng::AbstractRNG, s::Sampleable{Matrixvariate}, dims::Dims) =
    rand(rng, s, Array{Matrix{eltype(s)}}(undef, dims))

# single matrix-variate, must allocate one matrix
rand(rng::AbstractRNG, s::Sampleable{Matrixvariate}) =
    _rand!(rng, s, Matrix{eltype(s)}(undef, size(s)))

# single matrix-variate with pre-allocated matrix
function rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
               A::AbstractMatrix{<:Real})
    size(A) == size(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample size."))
    return _rand!(rng, s, A)
end

# pdf & logpdf

_pdf(d::MatrixDistribution, x::AbstractMatrix{T}) where {T<:Real} = exp(_logpdf(d, x))

"""
    logpdf(d::MatrixDistribution, AbstractMatrix)

Compute the logarithm of the probability density at the input matrix `x`.
"""
function logpdf(d::MatrixDistribution, x::AbstractMatrix{T}) where T<:Real
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf(d, x)
end

"""
    pdf(d::MatrixDistribution, x::AbstractArray)

Compute the probability density at the input matrix `x`.
"""
function pdf(d::MatrixDistribution, x::AbstractMatrix{T}) where T<:Real
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, x)
end

function _logpdf!(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M}) where M<:Matrix
    for i = 1:length(X)
        r[i] = logpdf(d, X[i])
    end
    return r
end

function _pdf!(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M}) where M<:Matrix
    for i = 1:length(X)
        r[i] = pdf(d, X[i])
    end
    return r
end

function logpdf!(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M}) where M<:Matrix
    length(X) == length(r) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf!(r, d, X)
end

function pdf!(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M}) where M<:Matrix
    length(X) == length(r) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf!(r, d, X)
end

function logpdf(d::MatrixDistribution, X::AbstractArray{M}) where M<:Matrix
    T = promote_type(partype(d), eltype(M))
    _logpdf!(Array{T}(undef, size(X)), d, X)
end

function pdf(d::MatrixDistribution, X::AbstractArray{M}) where M<:Matrix
    T = promote_type(partype(d), eltype(M))
    _pdf!(Array{T}(undef, size(X)), d, X)
end

"""
    _logpdf(d::MatrixDistribution, x::AbstractArray)

Evaluate logarithm of pdf value for a given sample `x`. This function need not perform dimension checking.
"""
_logpdf(d::MatrixDistribution, x::AbstractArray)

##### Specific distributions #####

for fname in ["wishart.jl", "inversewishart.jl"]
    include(joinpath("matrix", fname))
end
