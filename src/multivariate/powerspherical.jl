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

_rand(rng::AbstractRNG, d::PowerSpherical, x::AbstractVector) =
    _rand(rng, sampler(d), x)

#_logpdf
function _logpdf(d::PowerSpherical, x::AbstractArray)
    a, b = (length(d) - 1) / 2. + d.κ, (length(d) - 1) / 2.
    return log(2) * (-a-b) + lgamma(a+b) - lgamma(a) + b * log(π) + d.κ .* log(d.μ' * x .+ 1)
end

# entropy
function StatsBase.entropy(d::PowerSpherical)
    a, b = (length(d) - 1) / 2. + d.κ, (length(d) - 1) / 2.
    logC = -( (a+b) * log(2) + lgamma(a) + b * log(pi) - lgamma(a+b))
    return -(logC + d.κ * ( log(2) + digamma(a) - digamma(a+b)))
end