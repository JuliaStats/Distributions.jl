
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

# sampling

rand!(d::MatrixDistribution, A::AbstractArray{M}) where {M<:Matrix} = _rand!(sampler(d), A)

"""
    rand(d::MatrixDistribution, n)

Draw a sample matrix from the distribution `d`.
"""
rand(d::MatrixDistribution, n::Int) = _rand!(sampler(d), Vector{Matrix{eltype(d)}}(undef, n))

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
