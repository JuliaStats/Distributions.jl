"""
Uniform distribution on the hyperspherical unit ball in ``d`` dimensions.
"""
struct HyperSphericalUniform <: ContinuousMultivariateDistribution
    d :: Int
end

# basic properties
Base.length(s::HyperSphericalUniform) = s.d
Base.eltype(::HyperSphericalUniform) = Float32

# sampling
function _rand!(rng::AbstractRNG, ::HyperSphericalUniform, x::AbstractVector{T}) where T<:Real
    x .= randn(rng, T, length(x))
    normalize!(x)
end

function _rand(rng::AbstractRNG, ::HyperSphericalUniform, x::AbstractVector{T}) where T<:Real
    normalize!(randn(rng, T, length(x)))
end

function StatsBase.entropy(s::HyperSphericalUniform)
    halfd = s.d / 2
    return halfd * logπ - loggamma(halfd) + logtwo
end

function _logpdf(s::HyperSphericalUniform, x)
    return -entropy(s)
end
