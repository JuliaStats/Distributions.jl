
"""
    GeneralizedGaussian(α, μ, β)

The *Generalized Gaussian distribution*, more commonly known as the exponential
power or the generalized normal distribution, with scale 'α', location 'μ', and
shape 'β' has the probability density function

```math
f(x, \\mu, \\alpha, \\beta) = \\frac{\\beta}{2\\alpha\\Gamma(1/\\beta)} e^{-(\\frac{|x-\\mu|}{\\alpha})^\\beta} \\quad x \\in (-\\infty, +\\infty) , \\alpha > 0, \\beta > 0
```

The Generalized Gaussian (GGD) is a parametric distribution that incorporates the
Normal and Laplacian distributions as special cases where β = 1 and β = 2. As β → ∞,
the distribution approaches the Uniform distribution on (μ-α, μ+α).

```julia
GeneralizedGaussian()           # GGD with shape 2, scale 1, location 0, (the Normal distribution)
GeneralizedGaussian(loc,s,sh)   # GGD with location loc, scale s, and shape sh

params(d)                       # Get the parameters, i.e. (loc,s,sh,)
shape(d)                        # Get the shape parameter, sh
scale(d)                        # Get the scale parameter, s
location(d)                     # Get the location parameter, loc
```

External Links
 * [Generalized Gaussian on Wikipedia](http://en.wikipedia.org/wiki/Generalized_normal_distribution)
 """

struct GeneralizedGaussian{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    μ::T
    β::T

    GeneralizedGaussian{T}(μ::T,α::T,β::T) where {T} = (@check_args(GeneralizedGaussian, α > zero(α) && β > zero(β)); new{T}(μ,α,β))
end

GeneralizedGaussian(μ::T,α::T,β::T) where {T<:Real} = GeneralizedGaussian{T}(μ,α,β)
GeneralizedGaussian(μ::Integer,α::Integer,β::Integer) = GeneralizedGaussian(Float64(μ),Float64(α),Float64(β))
GeneralizedGaussian(β::Float64) = GeneralizedGaussian(0.0, 1.0, β)
GeneralizedGaussian() = GeneralizedGaussian(0.0, √2, 2.0) # approximate scale with unity std deviation and shape 2

#### Conversions

convert(::Type{GeneralizedGaussian{T}}, μ::S, α::S, β::S) where {T <: Real, S <: Real} = GeneralizedGaussian(T(μ),T(α),T(β))
convert(::Type{GeneralizedGaussian{T}}, d::GeneralizedGaussian{S}) where {T <: Real, S <: Real} = GeneralizedGaussian(T(d.μ), T(d.α), T(d.β))

@distr_support GeneralizedGaussian -Inf Inf


#### Parameters
params(d::GeneralizedGaussian) = (d.μ, d.α, d.β)
@inline partype(d::GeneralizedGaussian{T}) where {T<:Real} = T

location(d::GeneralizedGaussian) = d.μ
shape(d::GeneralizedGaussian) = d.β
scale(d::GeneralizedGaussian) = d.α


#### Statistics

mean(d::GeneralizedGaussian) = d.μ
median(d::GeneralizedGaussian) = d.μ
mode(d::GeneralizedGaussian) = d.μ

var(d::GeneralizedGaussian{T}) where {T<:Real} = (d.α^2) * ( gamma(3 / d.β) / gamma(1 / d.β) )
std(d::GeneralizedGaussian{T}) where {T<:Real} = (d.α) * sqrt( gamma(3 / d.β) / gamma(1 / d.β) )

skewness(d::GeneralizedGaussian{T}) where {T<:Real} = zero(T)
kurtosis(d::GeneralizedGaussian{T}) where {T<:Real} = (gamma(5 / d.β) * gamma(1 / d.β)) / (gamma(3 / d.β)^2) - 3

entropy(d::GeneralizedGaussian{T}) where {T<:Real} = (1 / d.β) - log( d.β / (2 * d.α * gamma(1 / d.β) ) )

#### Evaluation

function pdf(d::GeneralizedGaussian{T}, x::Real) where T<:Real
    (μ, α, β) = params(d)
    return ( β / ( 2 * α * gamma(1 / β) ) ) * exp( -( abs(x - μ) / α )^β )
end

function cdf(d::GeneralizedGaussian{T}, x::Real) where T<:Real

    #    To calculate the CDF, the incomplete gamma function is required. The CDF
    #   of the Gamma distribution provides this, with the necessary 1/Γ(a) normalization.

    (μ, α, β) = params(d)
    v = cdf(Gamma(1 / β, 1), (abs(x - μ) / α)^β) / 2
    return typeof(v)(1/2) + sign(x - μ) * v
end

#### Sampling
function rand(rng::AbstractRNG, d::GeneralizedGaussian)
    #   Sampling Procecdure from [1]
    #   [1]  Gonzalez-Farias, G., Molina, J. A. D., & Rodríguez-Dagnino, R. M. (2009).
    #   Efficiency of the approximated shape parameter estimator in the generalized
    #   Gaussian distribution. IEEE Transactions on Vehicular Technology, 58(8),
    #   4214-4223.

    # utilizing the sampler from the Gamma distribution.
    g = Gamma((1 / d.β), 1)

    # bernoulli random variable "b" with parameter (1/2).
    b = rand(1) .< 0.5
    b = convert(Float64, b[1]) * 2 - 1

    return d.μ + (1 / sqrt(d.α) ) * rand(rng, g)^(1 / d.β) * b
end
