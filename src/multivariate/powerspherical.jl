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

    function PowerSpherical(μ::Vector{T}, κ::T = one(T); check_args::Bool=true, normalize_μ::Bool = false) where {T <: Real}
        μ = normalize_μ ? normalize(μ) : μ
        @check_args PowerSpherical (κ, κ > zero(κ)) (μ, isunitvec(μ))
        new{eltype(μ)}(μ, κ)
    end
end

function PowerSpherical(μ::Vector{T}, κ::Real; check_args::Bool=true, normalize_μ::Bool = false) where {T<:Real}
    R = promote_type(T, eltype(κ))
    return PowerSpherical(convert(AbstractArray{R}, μ), convert(R, κ); check_args = check_args, normalize_μ = normalize_μ)
end

### Basic properties
Base.length(d::PowerSpherical) = length(d.μ)
Base.eltype(d::PowerSpherical) = eltype(d.μ)
meandir(d::PowerSpherical) = d.μ
concentration(d::PowerSpherical) = d.κ
insupport(d::PowerSpherical, x::AbstractVector{<:Real}) = length(x) == length(d) && isunitvec(x)

function sampler(d::PowerSpherical)
    dim = length(d)
    beta = (dim - 1) / 2
    return PowerSphericalSampler(
        d.μ,
        d.κ,
        Beta(beta + d.κ, beta; check_args=false), 
        HyperSphericalUniform(dim-1)
    )
end

#_logpdf
function _logpdf(d::PowerSpherical, x::AbstractArray)
    b = (length(d) - 1) // 2
    a = b + d.κ
    c = a + b

    return logtwo * (-a-b) + loggamma(c) - loggamma(a) + b * logπ + d.κ .* log1p(d.μ' * x)
end

# entropy
function StatsBase.entropy(d::PowerSpherical)
    b = (length(d) - 1) / convert(typeof(d.κ), 2)
    a = b + d.κ
    c = length(d) - 1 + d.κ

    _logtwo = convert(typeof(d.κ), logtwo)
    _logπ = convert(typeof(d.κ), logπ)

    logC = -(c * _logtwo + loggamma(a) + b * _logπ - loggamma(c))
    return -(logC + d.κ * ( _logtwo + digamma(a) - digamma(c)))
end

# analytical KL divergences
function kldivergence(p::PowerSpherical, q::HyperSphericalUniform)
    return -entropy(p) + entropy(q)
end

#TODO: add KL divergence for VonMissesFisher