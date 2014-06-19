
##### Generic methods #####

## domain

function insupport!{D<:MultivariateDistribution}(r::AbstractArray, d::Union(D,Type{D}), X::AbstractMatrix)
    n = length(r)
    size(X) == (length(d),n) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1:n
        @inbounds r[i] = insupport(d, view(X, :, i))
    end
    return r
end

insupport{D<:MultivariateDistribution}(d::Union(D,Type{D}), X::AbstractMatrix) = 
    insupport!(BitArray(size(X,2)), d, X)

## statistics

function cor(d::MultivariateDistribution)
    C = cov(d)
    n = size(C, 1)
    @assert size(C, 2) == n
    R = Array(Float64, n, n)

    for j = 1:n
        for i = 1:j-1
            @inbounds R[i, j] = R[j, i]
        end
        R[j, j] = 1.0
        for i = j+1:n
            @inbounds R[i, j] = C[i, j] / sqrt(C[i, i] * C[j, j])
        end
    end
    
    return R
end

# pdf and logpdf

_pdf(d::MultivariateDistribution, X::AbstractVector) = exp(_logpdf(d, X))

function logpdf(d::MultivariateDistribution, X::AbstractVector)
    length(X) == length(d) || 
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf(d, X)
end

function pdf(d::MultivariateDistribution, X::AbstractVector)
    length(X) == length(d) || 
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, X)
end

function _logpdf!(r::AbstractArray, d::MultivariateDistribution, X::DenseMatrix)
    for i in 1 : size(X,2)
        @inbounds r[i] = logpdf(d, view(X,:,i))
    end
    return r
end

function _pdf!(r::AbstractArray, d::MultivariateDistribution, X::DenseMatrix)
    for i in 1 : size(X,2)
        @inbounds r[i] = pdf(d, view(X,:,i))
    end
    return r
end

function logpdf!(r::AbstractArray, d::MultivariateDistribution, X::DenseMatrix)
    size(X) == (length(d), length(r)) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf!(r, d, X)
end

function pdf!(r::AbstractArray, d::MultivariateDistribution, X::DenseMatrix)
    size(X) == (length(d), length(r)) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf!(r, d, X)
end

function logpdf(d::MultivariateDistribution, X::DenseMatrix)
    size(X, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf!(Array(Float64, size(X,2)), d, X)
end

function pdf(d::MultivariateDistribution, X::DenseMatrix)
    size(X, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf!(Array(Float64, size(X,2)), d, X)
end

## log likelihood

function loglikelihood(d::MultivariateDistribution, X::DenseMatrix)
    ll = 0.0
    for i in 1:size(X, 2)
        ll += _logpdf(d, view(X,:,i))
    end
    return ll
end


##### Specific distributions #####

for fname in ["dirichlet.jl",
              "multinomial.jl",
              "mvnormal.jl", 
              "mvnormalcanon.jl",
              "mvtdist.jl",
              "vonmisesfisher.jl"]
    include(joinpath("multivariate", fname))
end
