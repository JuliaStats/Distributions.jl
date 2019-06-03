##### Generic methods #####

"""
    length(d::MultivariateDistribution) -> Int

Return the sample dimension of distribution `d`.
"""
length(d::MultivariateDistribution)

"""
    size(d::MultivariateDistribution)

Return the sample size of distribution `d`, *i.e* `(length(d),)`.
"""
size(d::MultivariateDistribution)

## sampling

"""
    rand!([rng::AbstractRNG,] d::MultivariateDistribution, x::AbstractArray)

Draw samples and output them to a pre-allocated array x. Here, x can be either
a vector of length `dim(d)` or a matrix with `dim(d)` rows.
"""
rand!(rng::AbstractRNG, d::MultivariateDistribution, x::AbstractArray)

# multivariate with pre-allocated array
function _rand!(rng::AbstractRNG, s::Sampleable{Multivariate}, m::AbstractMatrix)
    @boundscheck size(m, 1) == length(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    smp = sampler(s)
    for i in Base.OneTo(size(m,2))
        _rand!(rng, smp, view(m,:,i))
    end
    return m
end

# single multivariate with pre-allocated vector
function rand!(rng::AbstractRNG, s::Sampleable{Multivariate},
               v::AbstractVector{<:Real})
    @boundscheck length(v) == length(s) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand!(rng, s, v)
end

# multiple multivariates with pre-allocated array of maybe pre-allocated vectors
rand!(rng::AbstractRNG, s::Sampleable{Multivariate},
      X::AbstractArray{<:AbstractVector}) =
          @inbounds rand!(rng, s, X,
                          !all([isassigned(X,i) for i in eachindex(X)]) ||
                          !all(length.(X) .== length(s)))

function rand!(rng::AbstractRNG, s::Sampleable{Multivariate},
               X::AbstractArray{V}, allocate::Bool) where V <: AbstractVector
    smp = sampler(s)
    if allocate
        for i in eachindex(X)
            X[i] = _rand!(rng, smp, V(undef, size(s)))
        end
    else
        for x in X
            rand!(rng, smp, x)
        end
    end
    return X
end

# multiple multivariate, must allocate matrix or array of vectors
rand(s::Sampleable{Multivariate}, n::Int) = rand(GLOBAL_RNG, s, n)
rand(rng::AbstractRNG, s::Sampleable{Multivariate}, n::Int) =
    _rand!(rng, s, Matrix{eltype(s)}(undef, length(s), n))
rand(rng::AbstractRNG, s::Sampleable{Multivariate}, dims::Dims) =
    rand(rng, s, Array{Vector{eltype(s)}}(undef, dims), true)

# single multivariate, must allocate vector
rand(rng::AbstractRNG, s::Sampleable{Multivariate}) =
    _rand!(rng, s, Vector{eltype(s)}(undef, length(s)))

## domain

"""
    insupport(d::MultivariateDistribution, x::AbstractArray)

If ``x`` is a vector, it returns whether x is within the support of ``d``.
If ``x`` is a matrix, it returns whether every column in ``x`` is within the support of ``d``.
"""
insupport{D<:MultivariateDistribution}(d::Union{D, Type{D}}, x::AbstractArray)

function insupport!(r::AbstractArray, d::Union{D,Type{D}}, X::AbstractMatrix) where D<:MultivariateDistribution
    n = length(r)
    size(X) == (length(d),n) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1:n
        @inbounds r[i] = insupport(d, view(X, :, i))
    end
    return r
end

insupport(d::Union{D,Type{D}}, X::AbstractMatrix) where {D<:MultivariateDistribution} =
    insupport!(BitArray(undef, size(X,2)), d, X)

## statistics

"""
    mean(d::MultivariateDistribution)

Compute the mean vector of distribution `d`.
"""
mean(d::MultivariateDistribution)

"""
    var(d::MultivariateDistribution)

Compute the vector of element-wise variances for distribution `d`.
"""
var(d::MultivariateDistribution)

"""
    entropy(d::MultivariateDistribution)

Compute the entropy value of distribution `d`.
"""
entropy(d::MultivariateDistribution)

"""
    entropy(d::MultivariateDistribution, b::Real)

Compute the entropy value of distribution ``d``, w.r.t. a given base.
"""
entropy(d::MultivariateDistribution, b::Real) = entropy(d) / log(b)

"""
    cov(d::MultivariateDistribution)

Compute the covariance matrix for distribution `d`. (`cor` is provided based on `cov`).
"""
cov(d::MultivariateDistribution)

"""
    cor(d::MultivariateDistribution)

Computes the correlation matrix for distribution `d`.
"""
function cor(d::MultivariateDistribution)
    C = cov(d)
    n = size(C, 1)
    @assert size(C, 2) == n
    R = Matrix{eltype(C)}(undef, n, n)

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

"""
    pdf(d::MultivariateDistribution, x::AbstractArray)

Return the probability density of distribution `d` evaluated at `x`.

- If `x` is a vector, it returns the result as a scalar.
- If `x` is a matrix with n columns, it returns a vector `r` of length n, where `r[i]` corresponds
to `x[:,i]` (i.e. treating each column as a sample).

`pdf!(r, d, x)` will write the results to a pre-allocated array `r`.
"""
pdf(d::MultivariateDistribution, x::AbstractArray)

"""
    logpdf(d::MultivariateDistribution, x::AbstractArray)

Return the logarithm of probability density evaluated at `x`.

- If `x` is a vector, it returns the result as a scalar.
- If `x` is a matrix with n columns, it returns a vector `r` of length n, where `r[i]` corresponds to `x[:,i]`.

`logpdf!(r, d, x)` will write the results to a pre-allocated array `r`.
"""
logpdf(d::MultivariateDistribution, x::AbstractArray)

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

function _logpdf!(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix)
    for i in 1 : size(X,2)
        @inbounds r[i] = logpdf(d, view(X,:,i))
    end
    return r
end

function _pdf!(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix)
    for i in 1 : size(X,2)
        @inbounds r[i] = pdf(d, view(X,:,i))
    end
    return r
end

function logpdf!(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix)
    size(X) == (length(d), length(r)) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf!(r, d, X)
end

function pdf!(r::AbstractArray, d::MultivariateDistribution, X::AbstractMatrix)
    size(X) == (length(d), length(r)) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf!(r, d, X)
end

function logpdf(d::MultivariateDistribution, X::AbstractMatrix)
    size(X, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    T = promote_type(partype(d), eltype(X))
    _logpdf!(Vector{T}(undef, size(X,2)), d, X)
end

function pdf(d::MultivariateDistribution, X::AbstractMatrix)
    size(X, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    T = promote_type(partype(d), eltype(X))
    _pdf!(Vector{T}(undef, size(X,2)), d, X)
end

"""
    _logpdf{T<:Real}(d::MultivariateDistribution, x::AbstractArray)

Evaluate logarithm of pdf value for a given vector `x`. This function need not perform dimension checking.
Generally, one does not need to implement `pdf` (or `_pdf`) as fallback methods are provided in `src/multivariates.jl`.
"""
_logpdf(d::MultivariateDistribution, x::AbstractArray)

"""
    loglikelihood(d::MultivariateDistribution, x::AbstractMatrix)

The log-likelihood of distribution `d` w.r.t. all columns contained in matrix `x`.
"""
function loglikelihood(d::MultivariateDistribution, X::AbstractMatrix)
    size(X, 1) == length(d) || throw(DimensionMismatch("Inconsistent array dimensions."))
    return sum(i -> _logpdf(d, view(X, :, i)), 1:size(X, 2))
end

##### Specific distributions #####

for fname in ["dirichlet.jl",
              "multinomial.jl",
              "dirichletmultinomial.jl",
              "mvnormal.jl",
              "mvnormalcanon.jl",
              "mvlognormal.jl",
              "mvtdist.jl",
              "product.jl",
              "vonmisesfisher.jl"]
    include(joinpath("multivariate", fname))
end
