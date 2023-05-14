"""
    PGeneralizedGaussian(μ, α, p)

The *p-Generalized Gaussian distribution*, more commonly known as the exponential
power or the generalized normal distribution, with scale `α`, location `μ`, and
shape `p` has the probability density function

```math
f(x, \\mu, \\alpha, p) = \\frac{p}{2\\alpha\\Gamma(1/p)} e^{-(\\frac{|x-\\mu|}{\\alpha})^p} \\quad x \\in (-\\infty, +\\infty) , \\alpha > 0, p > 0
```

The p-Generalized Gaussian (GGD) is a parametric distribution that incorporates the
normal (`p = 2`) and Laplacian (`p = 1`) distributions as special cases.
As `p → ∞`, the distribution approaches the Uniform distribution on `[μ - α, μ + α]`.

```julia
PGeneralizedGaussian()           # GGD with location 0, scale √2, and shape 2 (the normal distribution)
PGeneralizedGaussian(μ, α, p)    # GGD with location μ, scale α, and shape p

params(d)                        # Get the parameters, i.e. (μ, α, p)
location(d)                      # Get the location parameter, μ
scale(d)                         # Get the scale parameter, α
shape(d)                         # Get the shape parameter, p
```

External Links
 * [Generalized Gaussian on Wikipedia](http://en.wikipedia.org/wiki/Generalized_normal_distribution)
 * [Reference implementation](https://www.researchgate.net/publication/254282790_Simulation_of_the_p-generalized_Gaussian_distribution)
 """
struct PGeneralizedGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    α::T
    p::T

    PGeneralizedGaussian{T}(μ::T, α::T, p::T) where {T<:Real} = new{T}(µ, α, p)
end

function PGeneralizedGaussian(μ::T, α::T, p::T; check_args::Bool=true) where {T<:Real}
    @check_args PGeneralizedGaussian (α, α > zero(α)) (p, p > zero(p))
    return PGeneralizedGaussian{T}(μ, α, p)
end
function PGeneralizedGaussian(μ::Real, α::Real, p::Real; check_args::Bool=true)
    return PGeneralizedGaussian(promote(μ, α, p)...; check_args=check_args)
end

"""
    PGeneralizedGaussian(p)

Build a p-generalized Gaussian with `μ=0.0, α=1.0`
"""
function PGeneralizedGaussian(p::Real; check_args::Bool=true)
    @check_args PGeneralizedGaussian (p, p > zero(p))
    return PGeneralizedGaussian{typeof(p)}(zero(p), oftype(p, 1), p)
end

"""
    PGeneralizedGaussian()

Builds a default p-generalized Gaussian with `μ=0.0, α=√2, p=2.0`, corresponding
to the normal distribution with `μ=0.0, σ=1.0`.
"""
PGeneralizedGaussian() = PGeneralizedGaussian{Float64}(0.0, √2, 2.0) # approximate scale with unity std deviation and shape 2

#### Conversions

function Base.convert(::Type{PGeneralizedGaussian{T}}, d::PGeneralizedGaussian) where {T<:Real}
    return PGeneralizedGaussian{T}(T(d.μ), T(d.α), T(d.p))
end
Base.convert(::Type{PGeneralizedGaussian{T}}, d::PGeneralizedGaussian{T}) where {T<:Real} = d

@distr_support PGeneralizedGaussian -Inf Inf


#### Parameters
partype(::PGeneralizedGaussian{T}) where {T<:Real} = T

params(d::PGeneralizedGaussian) = (d.μ, d.α, d.p)
location(d::PGeneralizedGaussian) = d.μ
shape(d::PGeneralizedGaussian) = d.p
scale(d::PGeneralizedGaussian) = d.α


#### Statistics

mean(d::PGeneralizedGaussian) = d.μ
median(d::PGeneralizedGaussian) = d.μ
mode(d::PGeneralizedGaussian) = d.μ

var(d::PGeneralizedGaussian) = d.α^2 * (gamma(3 / d.p) / gamma(1 / d.p))
std(d::PGeneralizedGaussian) = d.α * sqrt(gamma(3 / d.p) / gamma(1 / d.p))

skewness(d::PGeneralizedGaussian) = zero(d.p)
kurtosis(d::PGeneralizedGaussian) = gamma(5 / d.p) * gamma(1 / d.p) / gamma(3 / d.p)^2 - 3
entropy(d::PGeneralizedGaussian) = 1 / d.p - log(d.p / (2 * d.α * gamma(1 / d.p)))

#### Evaluation

function pdf(d::PGeneralizedGaussian, x::Real)
    μ, α, p = params(d)
    return (p / (2 * α * gamma(1 / p))) * exp(- (abs(x - μ) / α)^p)
end
function logpdf(d::PGeneralizedGaussian, x::Real)
    μ, α, p = params(d)
    return log(p / (2 * α)) - loggamma(1 / p) - (abs(x - μ) / α)^p
end

# To determine the CDF, the incomplete gamma function is required.
# The CDF of the Gamma distribution provides this, with the necessary 1/Γ(a) normalization.
function cdf(d::PGeneralizedGaussian, x::Real)
    μ, α, p = params(d)
    v = cdf(Gamma(inv(p), 1), (abs(x - μ) / α)^p)
    return (1 + copysign(v, x - μ)) / 2
end
function logcdf(d::PGeneralizedGaussian, x::Real)
    μ, α, p = params(d)
    Δ = x - μ
    logv = logcdf(Gamma(inv(p), 1), (abs(Δ) / α)^p)
    if Δ < 0
        return log1mexp(logv) - logtwo
    else
        return log1pexp(logv) - logtwo
    end
end

function quantile(d::PGeneralizedGaussian, q::Real)
    μ, α, p = params(d)
    inv_p = inv(p)
    r = 2 * q - 1
    z = α * quantile(Gamma(inv_p, 1), abs(r))^inv_p
    return μ + copysign(z, r)
end

#### Sampling

# The sampling procedure is implemented from from [1].
# [1]  Gonzalez-Farias, G., Molina, J. A. D., & Rodríguez-Dagnino, R. M. (2009).
# Efficiency of the approximated shape parameter estimator in the generalized
# Gaussian distribution. IEEE Transactions on Vehicular Technology, 58(8),
# 4214-4223.
function rand(rng::AbstractRNG, d::PGeneralizedGaussian)
    inv_p = inv(d.p)
    g = Gamma(inv_p, 1)
    z = d.α * rand(rng, g)^inv_p
    if rand(rng) < 0.5
        return d.μ - z
    else
        return d.μ + z
    end
end
