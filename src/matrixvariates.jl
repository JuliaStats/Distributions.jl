
##### Generic methods #####

# sampling

rand!{M<:Matrix}(d::MatrixDistribution, A::AbstractArray{M}) = _rand!(sampler(d), A)
rand(d::MatrixDistribution, n::Int) = _rand!(sampler(d), Array(Matrix{eltype(d)}, n))

# pdf & logpdf

_pdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T}) = exp(_logpdf(d, x))

function logpdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf(d, x)
end

function pdf{T<:Real}(d::MatrixDistribution, x::AbstractMatrix{T})
    size(x) == size(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, x)
end

function _logpdf!{M<:Matrix}(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M})
    for i = 1:length(X)
        r[i] = logpdf(r, X[i])
    end
    return r
end

function _pdf!{M<:Matrix}(r::AbstractArray, d::MatrixDistribution, X::AbstractArray{M})
    for i = 1:length(X)
        r[i] = pdf(r, X[i])
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
    _logpdf!(Array(T, size(X)), d, X)
end

function pdf{M<:Matrix}(d::MatrixDistribution, X::AbstractArray{M})
    T = promote_type(partype(d), eltype(M))
    _pdf!(Array(T, size(X)), d, X)
end


##### Specific distributions #####

for fname in ["wishart.jl", "inversewishart.jl"]
    include(joinpath("matrix", fname))
end
