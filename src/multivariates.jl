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
    _logpdf(d::MultivariateDistribution, x::AbstractVector)

Return the logarithm of the probability density of distribution `d` evaluated at `x`.

This function does not need to perform dimension checking.
"""
_logpdf(d::MultivariateDistribution, x::AbstractVector)

"""
    _pdf(d::MultivariateDistribution, x::AbstractVector)

Return the probability density of distribution `d` evaluated at `x`.

This function does not need to perform dimension checking. , this method falls
back to [`_logpdf`](@ref) and does not need to be implemented.
"""
_pdf(d::MultivariateDistribution, X::AbstractVector) = exp(_logpdf(d, X))

"""
    logpdf(d::MultivariateDistribution, x::AbstractVector)

Return the logarithm of the probability density of distribution `d` evaluated at `x`.

The default implementation performs dimension checking and falls back to [`_logpdf`](@ref).
"""
function logpdf(d::MultivariateDistribution, X::AbstractVector)
    length(X) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _logpdf(d, X)
end

"""
    pdf(d::MultivariateDistribution, x::AbstractVector)

Return the probability density of distribution `d` evaluated at `x`.

The default implementation performs dimension checking and falls back to [`_pdf`](@ref).
"""
function pdf(d::MultivariateDistribution, X::AbstractVector)
    length(X) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    _pdf(d, X)
end

"""
    loglikelihood(d::MultivariateDistribution, x::AbstractArray)

The log-likelihood of distribution `d` with respect to all samples contained in array `x`.

Here, `x` can be a vector of length `dim(d)`, a matrix with `dim(d)` rows, or an array of
vectors of length `dim(d)`.
"""
loglikelihood(d::MultivariateDistribution, X::AbstractVector{<:Real}) = logpdf(d, X)
function loglikelihood(d::MultivariateDistribution, X::AbstractMatrix{<:Real})
    size(X, 1) == length(d) || throw(DimensionMismatch("Inconsistent array dimensions."))
    return sum(i -> _logpdf(d, view(X, :, i)), 1:size(X, 2))
end
function loglikelihood(d::MultivariateDistribution, X::AbstractArray{<:AbstractVector})
    return sum(x -> logpdf(d, x), X)
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
