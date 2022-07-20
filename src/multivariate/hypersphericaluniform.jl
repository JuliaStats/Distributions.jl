"""
Uniform distribution on the hyperspherical unit ball in ``d`` dimensions.
"""
struct HyperSphericalUniform <: ContinuousMultivariateDistribution
    d :: Int
end

# basic properties
Base.length(s::HyperSphericalUniform) = s.d
insupport(s::HyperSphericalUniform, x::AbstractVector{<:Real}) = length(x) == length(s) && isunitvec(x)

# sampling
function _rand!(rng::AbstractRNG, ::HyperSphericalUniform, x::AbstractVector{<:Real})
    randn!(rng, x)
    normalize!(x)
end

function StatsBase.entropy(s::HyperSphericalUniform)
    halfd = s.d / 2
    return halfd * logπ - loggamma(halfd) + logtwo
end

function _logpdf(s::HyperSphericalUniform, x::AbstractVector{<:Real})
    y = entropy(s)
    return insupport(s, x) ? -y : oftype(y, -Inf)
end