
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

# multivariate with pre-allocated 3D array
function _rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
                m::AbstractArray{<:Real, 3})
    @boundscheck (size(m, 1), size(m, 2)) == (size(s, 1), size(s, 2)) ||
        throw(DimensionMismatch("Output size inconsistent with matrix size."))
    smp = sampler(s)
    for i in Base.OneTo(size(m,3))
        _rand!(rng, smp, view(m,:,:,i))
    end
    return m
end

# multiple matrix-variates with pre-allocated array of maybe pre-allocated matrices
rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
      X::AbstractArray{<:AbstractMatrix}) =
          @inbounds rand!(rng, s, X,
                          !all([isassigned(X,i) for i in eachindex(X)]) ||
                          (sz = size(s); !all(size(x) == sz for x in X)))

function rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
               X::AbstractArray{M}, allocate::Bool) where M <: AbstractMatrix
    smp = sampler(s)
    if allocate
        for i in eachindex(X)
            X[i] = _rand!(rng, smp, M(undef, size(s)))
        end
    else
        for x in X
            rand!(rng, smp, x)
        end
    end
    return X
end

# multiple matrix-variates, must allocate array of arrays
rand(rng::AbstractRNG, s::Sampleable{Matrixvariate}, dims::Dims) =
    rand!(rng, s, Array{Matrix{eltype(s)}}(undef, dims), true)

# single matrix-variate, must allocate one matrix
rand(rng::AbstractRNG, s::Sampleable{Matrixvariate}) =
    _rand!(rng, s, Matrix{eltype(s)}(undef, size(s)))

# single matrix-variate with pre-allocated matrix
function rand!(rng::AbstractRNG, s::Sampleable{Matrixvariate},
               A::AbstractMatrix{<:Real})
    @boundscheck size(A) == size(s) ||
        throw(DimensionMismatch("Output size inconsistent with matrix size."))
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
