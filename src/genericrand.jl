### Generic rand methods

# univariate

function _rand!(s::Sampleable{Univariate}, A::AbstractArray)
    for i in 1:length(A)
        @inbounds A[i] = rand(s)
    end
    return A
end
rand!(s::Sampleable{Univariate}, A::AbstractArray) = _rand!(s, A)

rand{S<:ValueSupport}(s::Sampleable{Univariate,S}, shp::Union(Int,(Int...))) = 
    _rand!(s, Array(eltype(S), shp))

# multivariate

function _rand!(s::Sampleable{Multivariate}, A::DenseMatrix)
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

function rand!(s::Sampleable{Multivariate}, A::DenseMatrix)
    size(A,1) == length(s) || 
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand!(s, A)
end

rand{S<:ValueSupport}(s::Sampleable{Multivariate,S}) = 
    _rand!(s, Array(eltype(S), length(s)))

rand{S<:ValueSupport}(s::Sampleable{Multivariate,S}, n::Int) = 
    _rand!(s, Array(eltype(S), length(s), n))


# matrix-variate

function _rand!{M<:Matrix}(s::Sampleable{Matrixvariate}, X::AbstractArray{M})
    for i in 1:length(X)
        X[i] = rand(s)
    end
    return X
end

rand!{M<:Matrix}(s::Sampleable{Matrixvariate}, X::AbstractArray{M}) = 
    _rand!(s, X)

rand{S<:ValueSupport}(s::Sampleable{Matrixvariate,S}, n::Int) =
    rand!(s, Array(Matrix{eltype(S)}, n))


# sampler

# one can specialize this function to provide more efficient samplers
# for certain distributions
sampler(d::Distribution) = d

rand!(s::UnivariateDistribution, A::AbstractArray) = _rand!(sampler(s), A)
rand!(s::MultivariateDistribution, A::DenseMatrix) = _rand!(sampler(s), A)
rand!{M<:Matrix}(s::MatrixDistribution, A::AbstractArray{M}) = _rand!(sampler(s), A)

