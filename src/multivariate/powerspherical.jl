"""
The Power Spherical distribution is useful as a replacement for the von Mises-Fisher distribution.

The probability density function of the Power Spherical distribution parameterized by the mean direction ``\\mu`` and the concentration parameter ``\\kappa`` is given by
    
```math
p_{X}(x ; \\mu, \\kappa)= \\left\\{2^{\\alpha+\\beta} \\pi^{\\beta} \\frac{\\Gamma(\\alpha)}{\\Gamma(\\alpha+\\beta)}\\right\\}^{-1}\\left(1+\\mu^{\\top} x\\right)^{\\kappa}
```
"""
struct PowerSpherical{T <: Real} <: ContinuousMultivariateDistribution
    μ::Vector{T}
    κ::T

    PowerSpherical(μ) =
        new{eltype(μ)}(normalize(μ) : μ, norm(μ))

    PowerSpherical(μ, κ; normalize_μ = true) =
        new{eltype(μ)}(normalize_μ ? normalize(μ) : μ, κ)
end

### Basic properties
Base.length(d::PowerSpherical) = length(d.μ)
Base.eltype(d::PowerSpherical) = eltype(d.μ)
meandir(d::PowerSpherical) = d.μ
concentration(d::PowerSpherical) = d.κ
insupport(::PowerSpherical, x::AbstractVector{T}) where {T<:Real} = isunitvec(x)

function sampler(d::PowerSpherical)
    dim = length(d)
    return PowerSphericalSampler(
        d.μ,
        d.κ,
        dim, 
        Beta((dim - 1) / 2. + d.κ, (dim - 1) / 2.), 
        HyperSphericalUniform(dim-1)
    )
end

_rand!(rng::AbstractRNG, d::PowerSpherical, x::AbstractVector) =
    _rand!(rng, sampler(d), x)

rand(rng::AbstractRNG, d::PowerSpherical) =
    rand(rng, sampler(d))

#_logpdf
function _logpdf(d::PowerSpherical, x::AbstractArray)
    b = (length(d) - 1) // 2
    a = b + d.κ
    c = a + b

    return logtwo * (-a-b) + loggamma(c) - loggamma(a) + b * logπ + d.κ .* log1p(d.μ' * x)
end


# entropy
function StatsBase.entropy(d::PowerSpherical)
    b = (length(d) - 1) / 2
    a = b + d.κ
    c = a + b

    logC = -(c * logtwo + loggamma(a) + b * logπ - loggamma(c))
    return -(logC + d.κ * ( logtwo + digamma(a) - digamma(c)))
end

# analytical KL divergences
function kldivergence(p::PowerSpherical, q::HyperSphericalUniform)
    return -entropy(p) + entropy(q)
end

#TODO: add KL divergence for VonMissesFisher