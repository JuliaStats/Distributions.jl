struct MvDiscreteNonParametric{T <: Real,P <: Real,Ts <:  ArrayOfSimilarArrays{T},Ps <: AbstractVector{P}} <: DiscreteMultivariateDistribution

    support::Ts
    p::Ps

    function MvDiscreteNonParametric{T,P,Ts,Ps}(support::Ts,
        p::Ps) where {T <: Real,P <: Real,Ts <: AbstractVector{<:AbstractVector{T}},Ps <: AbstractVector{P}}

        length(support) == length(p) || error("length of `support` and `p` must be equal")
        isprobvec(p) || error("`p` must be a probability vector")
        allunique(support) || error("`support` must contain only unique value")
        new{T,P,Ts,Ps}(support, p)
    end
end

"""
    MvDiscreteNonParametric(
        support::AbstractVector,
        p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
    )

Construct a multivariate discrete nonparametric probability distribution with `support` and corresponding
probabilities `p`. If the probability vector argument is not passed, then
equal probability is assigned to each entry in the support.

# Examples
```julia
using ArraysOfArrays
# rows correspond to samples
μ = MvDiscreteNonParametric(nestedview(rand(7,3)'))

# columns correspond to samples
ν = MvDiscreteNonParametric(nestedview(rand(7,3)))
```
"""
function MvDiscreteNonParametric(
    support::AbstractVector{<:AbstractVector{<:Real}},
    p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
)
    return MvDiscreteNonParametric{eltype(eltype(support)),eltype(p),typeof(ArrayOfSimilarArrays(support)),typeof(p)}(
        ArrayOfSimilarArrays(support), p)
end

"""
    MvDiscreteNonParametric(
        support::Matrix{<:Real},
        p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)
    )

Construct a multivariate discrete nonparametric probability distribution
from a matrix as `support` where each row is a sample, and corresponding
probabilities `p`. If the probability vector argument is not passed, then
equal probability is assigned to each entry in the support.

# Examples
```julia
# the rows correspond to the samples
using LinearAlgebra
μ = MvDiscreteNonParametric(rand(10,3), normalize!(rand(10),1))
```
"""
function MvDiscreteNonParametric(
    support::Matrix{<:Real},
    p::AbstractVector{<:Real}=fill(inv(size(support)[1]), size(support)[1])
)
    return MvDiscreteNonParametric(nestedview(support'), p)
end

Base.eltype(::Type{<:MvDiscreteNonParametric{T}}) where T = T

"""
    support(d::MvDiscreteNonParametric)
Get a sorted AbstractVector defining the support of `d`.
"""
Distributions.support(d::MvDiscreteNonParametric) = d.support

"""
    probs(d::MvDiscreteNonParametric)
Get the vector of probabilities associated with the support of `d`.
"""
Distributions.probs(d::MvDiscreteNonParametric) = d.p


# It would be more intuitive if length was the
# 
"""
    length(d::MvDiscreteNonParametric)
Retunrs the dimension of the mass points (samples).
It corresponds to `innersize(d.support)[1]`,
where `innersize` is a function from from ArraysOfArrays.jl.
"""
Base.length(d::MvDiscreteNonParametric) = innersize(d.support)[1]

"""
    size(d::MvDiscreteNonParametric)
Returns the size of the support as a tuple where
the first value is the number of points (samples)
and the second is the dimension of the samples (e.g ℝⁿ).
It corresponds to `size(flatview(d.support)')`
where `flatview` is a function from from ArraysOfArrays.jl
that turns an Array of Arrays into a matrix.
"""
Base.size(d::MvDiscreteNonParametric) = size(flatview(d.support)')

"""
    empiricalmeasure(
        support::AbstractVector,
        probs::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
    )

Construct a finite discrete probability measure with `support` and corresponding
`probabilities`. If the probability vector argument is not passed, then
equal probability is assigned to each entry in the support.

# Examples
```julia
using ArraysOfArrays
# rows correspond to samples
μ = empiricalmeasure(nestedview(rand(7,3)'), normalize!(rand(10),1))

# columns correspond to samples, each with equal probability
ν = empiricalmeasure(nestedview(rand(3,12)))
```

!!! note
    If `support` is a 1D vector, the constructed measure will be sorted,
    e.g. for `mu = empiricalmeasure([3, 1, 2],[0.5, 0.2, 0.3])`, then
    `mu.support` will be `[1, 2, 3]` and `mu.p` will be `[0.2, 0.3, 0.5]`.
    Also, avoid passing 1D distributions as `RowVecs(rand(3))` or `[[1],[3],[4]]`,
    since this will be dispatched to the multivariate case instead
    of the univariate case for which the algorithm is more efficient.

"""
function empiricalmeasure(
    support::AbstractVector{<:Real},
    p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
    )
    return DiscreteNonParametric(support, p)
end
function empiricalmeasure(
    support::AbstractVector{<:AbstractVector{<:Real}},
    p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
    )
    return MvDiscreteNonParametric(support, p)
    # return MvDiscreteNonParametric{eltype(eltype(support)),eltype(p),typeof(support),typeof(p)}(support, p)
end

"""
    empiricalmeasure(
        support::Matrix{<:Real},
        probs::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
    )
Construct a multivariate empirical measure from a matrix
as `support` and sample by rows.
# Examples
```julia
using ArraysOfArrays
# the rows correspond to the samples
μ = empiricalmeasure(rand(7,3), normalize!(rand(10),1))
```
"""
function empiricalmeasure(
    support::Matrix{<:Real},
    p::AbstractVector{<:Real}=fill(inv(size(support)[1]), size(support)[1])
)
    return MvDiscreteNonParametric(nestedview(support'), p)
end


function _logpdf(d::MvDiscreteNonParametric, x::AbstractVector{T}) where T <: Real
    s = support(d)
    p = probs(d)
    for i in 1:length(p)
        if s[i] == x
            return log(p[i])
        end
    end
    return log(zero(eltype(p)))
end


function mean(d::MvDiscreteNonParametric)
    return StatsBase.mean(d.support, weights(d.p))
end

function var(d::MvDiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return StatsBase.var(x, Weights(p, one(eltype(p))), corrected=false)
end

function cov(d::MvDiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return cov(x, Weights(p, one(eltype(p))), corrected=false)
end

entropy(d::MvDiscreteNonParametric) = entropy(probs(d))
entropy(d::MvDiscreteNonParametric, b::Real) = entropy(probs(d), b)
