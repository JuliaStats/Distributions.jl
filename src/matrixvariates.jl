
##### Generic methods #####

"""
    size(d::MatrixDistribution)

Return the size of each sample from distribution `d`.
"""
size(d::MatrixDistribution) = throw(MethodError(size, (d,)))

"""
    length(d::MatrixDistribution)

The length (*i.e* number of elements) of each sample from the distribution `d`.
"""
Base.length(d::MatrixDistribution) = prod(size(d))

"""
    mean(d::MatrixDistribution)

Return the mean matrix of `d`.
"""
mean(d::MatrixDistribution) = throw(MethodError(size, (d,)))

# sampling

rand!{M<:Matrix}(d::MatrixDistribution, A::AbstractArray{M}) = _rand!(sampler(d), A)

"""
    rand(d::MatrixDistribution, [n])

Draw a sample matrix from the distribution `d`.
"""
rand(d::MatrixDistribution, n::Int=1) = _rand!(sampler(d), Vector{Matrix{eltype(d)}}(n))

# pdf & logpdf

_pdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T}) = exp(_logpdf(d, x))

"""
    logpdf(d::MatrixDistribution, AbstractMatrix)

Compute the logarithm of the probability density at the input matrix `x`.
"""
function logpdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf(d, x)
end

"""
    pdf(d::MatrixDistribution, x::AbstractArray)

Compute the probability density at the input matrix `x`.
"""
function pdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, x)
end

function _logpdf!{M<:Matrix}(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M})
    for i = 1:length(X)
        r[i] = logpdf(d, X[i])
    end
    return r
end

function _pdf!{M<:Matrix}(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M})
    for i = 1:length(X)
        r[i] = pdf(d, X[i])
    end
    return r
end

function logpdf!{M<:Matrix}(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M})
    length(X) == length(r) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf!(r, d, X)
end

function pdf!{M<:Matrix}(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M})
    length(X) == length(r) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf!(r, d, X)
end

function logpdf{M<:Matrix}(d::MatrixDistribution, X::AbstractArray{M})
    T = promote_type(partype(d), eltype(M))
    _logpdf!(Array{T}(size(X)), d, X)
end

function pdf{M<:Matrix}(d::MatrixDistribution, X::AbstractArray{M})
    T = promote_type(partype(d), eltype(M))
    _pdf!(Array{T}(size(X)), d, X)
end

"""
    _logpdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})

Evaluate logarithm of pdf value for a given sample `x`. This function need not perform dimension checking.
"""
_logpdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T}) = throw(MethodError(_logpdf, (d, x)))

##### Specific distributions #####

for fname in ["wishart.jl", "inversewishart.jl"]
    include(joinpath("matrix", fname))
end
