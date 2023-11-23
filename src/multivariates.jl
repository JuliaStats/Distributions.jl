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

# multiple multivariate, must allocate matrix
# TODO: inconsistency with other `ArrayLikeVariate`s and `rand(s, (n,))` - maybe remove?
rand(rng::AbstractRNG, s::Sampleable{Multivariate}, n::Int) =
    @inbounds rand!(rng, sampler(s), Matrix{eltype(s)}(undef, length(s), n))
rand(rng::AbstractRNG, s::Sampleable{Multivariate,Continuous}, n::Int) =
    @inbounds rand!(rng, sampler(s), Matrix{float(eltype(s))}(undef, length(s), n))

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

##### Specific distributions #####

for fname in ["dirichlet.jl",
              "multinomial.jl",
              "dirichletmultinomial.jl",
              "jointorderstatistics.jl",
              "mvnormal.jl",
              "mvnormalcanon.jl",
              "mvlogitnormal.jl",
              "mvlognormal.jl",
              "mvtdist.jl",
              "product.jl", # deprecated
              "vonmisesfisher.jl"]
    include(joinpath("multivariate", fname))
end
