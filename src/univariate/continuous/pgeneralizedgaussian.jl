
"""
    PGeneralizedGaussian(α, μ, p)

The *p-Generalized Gaussian distribution*, more commonly known as the exponential
power or the generalized normal distribution, with scale 'α', location 'μ', and
shape 'p' has the probability density function

```math
f(x, \\mu, \\alpha, p) = \\frac{p}{2\\alpha\\Gamma(1/p)} e^{-(\\frac{|x-\\mu|}{\\alpha})^p} \\quad x \\in (-\\infty, +\\infty) , \\alpha > 0, p > 0
```

The p-gGneralized Gaussian (GGD) is a parametric distribution that incorporates the
Normal and Laplacian distributions as special cases where p = 1 and p = 2. As p → ∞,
the distribution approaches the Uniform distribution on (μ-α, μ+α).

```julia
PGeneralizedGaussian()           # GGD with shape 2, scale 1, location 0, (the Normal distribution)
PGeneralizedGaussian(loc,s,sh)   # GGD with location loc, scale s, and shape sh

params(d)                       # Get the parameters, i.e. (loc,s,sh,)
shape(d)                        # Get the shape parameter, sh
scale(d)                        # Get the scale parameter, s
location(d)                     # Get the location parameter, loc
```

External Links
 * [Generalized Gaussian on Wikipedia](http://en.wikipedia.org/wiki/Generalized_normal_distribution)
 * [R Package reference](https://www.researchgate.net/publication/254282790_Simulation_of_the_p-generalized_Gaussian_distribution)
 """

struct PGeneralizedGaussian{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    μ::T
    p::T

    PGeneralizedGaussian{T}(μ::T,α::T,p::T) where {T} = (@check_args(PGeneralizedGaussian, α > zero(α) && p > zero(p)); new{T}(μ,α,p))
end

PGeneralizedGaussian(μ::T,α::T,p::T) where {T<:Real} = PGeneralizedGaussian{T}(μ,α,p)
PGeneralizedGaussian(μ::Integer,α::Integer,p::Integer) = PGeneralizedGaussian(Float64(μ),Float64(α),Float64(p))
PGeneralizedGaussian(p::Float64) = PGeneralizedGaussian(0.0, 1.0, p)
PGeneralizedGaussian() = PGeneralizedGaussian(0.0, √2, 2.0) # approximate scale with unity std deviation and shape 2

#### Conversions

convert(::Type{PGeneralizedGaussian{T}}, μ::S, α::S, p::S) where {T <: Real, S <: Real} = PGeneralizedGaussian(T(μ),T(α),T(p))
convert(::Type{PGeneralizedGaussian{T}}, d::PGeneralizedGaussian{S}) where {T <: Real, S <: Real} = PGeneralizedGaussian(T(d.μ), T(d.α), T(d.p))

@distr_support PGeneralizedGaussian -Inf Inf


#### Parameters
params(d::PGeneralizedGaussian) = (d.μ, d.α, d.p)
@inline partype(d::PGeneralizedGaussian{T}) where {T<:Real} = T

location(d::PGeneralizedGaussian) = d.μ
shape(d::PGeneralizedGaussian) = d.p
scale(d::PGeneralizedGaussian) = d.α


#### Statistics

mean(d::PGeneralizedGaussian) = d.μ
median(d::PGeneralizedGaussian) = d.μ
mode(d::PGeneralizedGaussian) = d.μ

var(d::PGeneralizedGaussian{T}) where {T<:Real} = (d.α^2) * ( gamma(3 / d.p) / gamma(1 / d.p) )
std(d::PGeneralizedGaussian{T}) where {T<:Real} = (d.α) * sqrt( gamma(3 / d.p) / gamma(1 / d.p) )

skewness(d::PGeneralizedGaussian{T}) where {T<:Real} = zero(T)
kurtosis(d::PGeneralizedGaussian{T}) where {T<:Real} = (gamma(5 / d.p) * gamma(1 / d.p)) / (gamma(3 / d.p)^2) - 3

# Skewness from Nadarajah, Saralees (September 2005).  "A generalized normal distribution".
# Journal of Applied Statistics. 32 (7): 685–694
entropy(d::PGeneralizedGaussian{T}) where {T<:Real} = (1 / d.p) - log( d.p / (2 * d.α * gamma(1 / d.p) ) )

#### Evaluation

function pdf(d::PGeneralizedGaussian{T}, x::Real) where T<:Real
    (μ, α, p) = params(d)
    return ( p / ( 2 * α * gamma(1 / p) ) ) * exp( -( abs(x - μ) / α )^p )
end

function cdf(d::PGeneralizedGaussian{T}, x::Real) where T<:Real

    #    To calculate the CDF, the incomplete gamma function is required. The CDF
    #   of the Gamma distribution provides this, with the necessary 1/Γ(a) normalization.

    (μ, α, p) = params(d)
    v = cdf(Gamma(1 / p, 1), (abs(x - μ) / α)^p) / 2
    return typeof(v)(1/2) + sign(x - μ) * v
end

#### Sampling
function rand(rng::AbstractRNG, d::PGeneralizedGaussian)
    #   Sampling Procecdure from [1]
    #   [1]  Gonzalez-Farias, G., Molina, J. A. D., & Rodríguez-Dagnino, R. M. (2009).
    #   Efficiency of the approximated shape parameter estimator in the generalized
    #   Gaussian distribution. IEEE Transactions on Vehicular Technology, 58(8),
    #   4214-4223.

    # utilizing the sampler from the Gamma distribution.
    g = Gamma(inv(d.p), 1)

    # bernoulli random variable "b" with parameter (1/2).
    b = Float64(rand(Bernoulli()))

    return d.μ + inv(sqrt(d.α)) * rand(rng, g)^inv(d.p) * b
end
