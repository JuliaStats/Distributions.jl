# struct MvDiscreteNonParametric{T <: Real,P <: Real,Ts <:  ArrayOfSimilarArrays{T},Ps <: AbstractVector{P}} <: DiscreteMultivariateDistribution
#     support::Ts
#     p::Ps
#     function MvDiscreteNonParametric{T,P,Ts,Ps}(support::Ts,
#         p::Ps) where {T <: Real,P <: Real,Ts <: AbstractVector{<:AbstractVector{T}},Ps <: AbstractVector{P}}
#         length(support) == length(p) || error("length of `support` and `p` must be equal")
#         isprobvec(p) || error("`p` must be a probability vector")
#         allunique(support) || error("`support` must contain only unique value")
#         new{T,P,Ts,Ps}(support, p)
#     end
# end
# """
#     MvDiscreteNonParametric(
#         support::AbstractVector,
#         p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
#     )
# Construct a multivariate discrete nonparametric probability distribution with `support` and corresponding
# probabilities `p`. If the probability vector argument is not passed, then
# equal probability is assigned to each entry in the support.
# # Examples
# ```julia
# using ArraysOfArrays
# # rows correspond to samples
# μ = MvDiscreteNonParametric(nestedview(rand(7,3)'))
# # columns correspond to samples
# ν = MvDiscreteNonParametric(nestedview(rand(7,3)))
# ```
# """
# function MvDiscreteNonParametric(
#     support::AbstractVector{<:AbstractVector{<:Real}},
#     p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
# )
#     return MvDiscreteNonParametric{eltype(eltype(support)),eltype(p),typeof(ArrayOfSimilarArrays(support)),typeof(p)}(
#         ArrayOfSimilarArrays(support), p)
# end
# """
#     MvDiscreteNonParametric(
#         support::Matrix{<:Real},
#         p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)
#     )
# Construct a multivariate discrete nonparametric probability distribution
# from a matrix as `support` where each row is a sample, and corresponding
# probabilities `p`. If the probability vector argument is not passed, then
# equal probability is assigned to each entry in the support.
# # Examples
# ```julia
# # the rows correspond to the samples
# using LinearAlgebra
# μ = MvDiscreteNonParametric(rand(10,3), normalize!(rand(10),1))
# ```
# """
# function MvDiscreteNonParametric(
#     support::Matrix{<:Real},
#     p::AbstractVector{<:Real}=fill(inv(size(support)[1]), size(support)[1])
# )
#     return MvDiscreteNonParametric(nestedview(support'), p)
# end
# Base.eltype(::Type{<:MvDiscreteNonParametric{T}}) where T = T



# struct GeneralDiscreteNonParametric{VF,T,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}} <: Distribution{VF,Discrete}

#     support::Ts
#     p::Ps

#     function GeneralDiscreteNonParametric{VF,T,P,Ts,Ps}(support::Ts,
#         p::Ps) where {T,P <: Real,Ts <: AbstractVector{<:AbstractVector{T}},Ps <: AbstractVector{P}}

#         length(support) == length(p) || error("length of `support` and `p` must be equal")
#         isprobvec(p) || error("`p` must be a probability vector")
#         allunique(support) || error("`support` must contain only unique value")
#         new{VF,T,P,Ts,Ps}(support, p)
#     end
# end

# const MvDiscreteNonParametric{T<:AbstractVector{<:Real},
#     P<:Real,Ts<:AbstractVector{T},
#     Ps<:AbstractVector{P}} = GeneralDiscreteNonParametric{Multivariate,T,P,Ts,Ps}

struct MvDiscreteNonParametric{T <: Real,P <: Real,
    Ts <: AbstractVector{<: AbstractVector{T}},Ps <: AbstractVector{P}} <: DiscreteMultivariateDistribution
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


# struct MvDiscreteNonParametric{VF,T,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}} <: Distribution{VF,Discrete}

#     support::Ts
#     p::Ps

#     function MvDiscreteNonParametric{T,P,Ts,Ps}(support::Ts,
#         p::Ps) where {T <: Real,P <: Real,Ts <: AbstractVector{<:AbstractVector{T}},Ps <: AbstractVector{P}}

#         length(support) == length(p) || error("length of `support` and `p` must be equal")
#         isprobvec(p) || error("`p` must be a probability vector")
#         allunique(support) || error("`support` must contain only unique value")
#         new{T,P,Ts,Ps}(support, p)
#     end
# end
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
    support::AbstractArray{<:AbstractVector{<:Real}},
    p::AbstractVector{<:Real}=fill(inv(length(support)), length(support)),
)
    return MvDiscreteNonParametric{eltype(eltype(support)),eltype(p),typeof(support),typeof(p)}(support, p)
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
support(d::MvDiscreteNonParametric) = d.support

"""
    probs(d::MvDiscreteNonParametric)
Get the vector of probabilities associated with the support of `d`.
"""
probs(d::MvDiscreteNonParametric) = d.p


# It would be more intuitive if length was the
#
Base.length(d::MvDiscreteNonParametric) = length(first(d.support))
Base.size(d::MvDiscreteNonParametric) = (length(d), length(d.support))

function _rand!(rng::AbstractRNG, d::MvDiscreteNonParametric, x::AbstractVector{T}) where T <: Real

    length(x) == length(d) || throw(DimensionMismatch("Invalid argument dimension."))
    s = d.support
    p = d.p

    n = length(p)
    draw = Base.rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i += 1]
    end
    copyto!(x, s[i])
    return x
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
