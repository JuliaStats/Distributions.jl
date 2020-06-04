"""
    PGeneralizedGaussian(α, μ, p)

The *p-Generalized Gaussian distribution*, more commonly known as the exponential
power or the generalized normal distribution, with scale `α`, location `μ`, and
shape `p` has the probability density function

```math
f(x, \\mu, \\alpha, p) = \\frac{p}{2\\alpha\\Gamma(1/p)} e^{-(\\frac{|x-\\mu|}{\\alpha})^p} \\quad x \\in (-\\infty, +\\infty) , \\alpha > 0, p > 0
```

The p-Generalized Gaussian (GGD) is a parametric distribution that incorporates the
Normal and Laplacian distributions as special cases where `p = 1` and `p = 2`. As `p → ∞`,
the distribution approaches the Uniform distribution on `[μ-α, μ+α]`.

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
 * [Reference implementation paper](https://www.researchgate.net/publication/254282790_Simulation_of_the_p-generalized_Gaussian_distribution)
 """
struct PGeneralizedGaussian{T1<:Real, T2<:Real, T3<:Real} <: ContinuousUnivariateDistribution
    μ::T1
    α::T2
    p::T3
    PGeneralizedGaussian{T1,T2,T3}(μ::T1,α::T2,p::T3) where {T1<:Real, T2<:Real, T3<:Real} = new{T1,T2,T3}(µ, α, p)
end

function PGeneralizedGaussian(μ::T1,α::T2,p::T3; check_args=true) where {T1<:Real, T2<:Real, T3<:Real}
    check_args && @check_args(PGeneralizedGaussian, α > zero(α) && p > zero(p))
    return PGeneralizedGaussian{T1,T2,T3}(μ,α,p)
end

"""
    PGeneralizedGaussian(p)

Builds a p-generalized Gaussian with `μ=0.0, α=1.0`
"""
PGeneralizedGaussian(p::T) where {T<:Real} = PGeneralizedGaussian(zero(T), one(T), p)

"""
    PGeneralizedGaussian()

Builds a default p-generalized Gaussian with `μ=0.0, α=√2, p=2.0`, corresponding
to the normal distribution with `μ=0.0, σ=1.0`.
"""
PGeneralizedGaussian() = PGeneralizedGaussian(0.0, √2, 2.0, check_args=false) # approximate scale with unity std deviation and shape 2

#### Conversions

convert(::Type{PGeneralizedGaussian{T1,T2,T3}}, μ::S1, α::S2, p::S3) where {T1 <: Real, T2 <: Real, T3 <:Real, S1 <: Real, S2 <: Real, S3 <: Real} = PGeneralizedGaussian(T1(μ),T2(α),T3(p))
function convert(::Type{PGeneralizedGaussian{T1,T2,T3}}, d::PGeneralizedGaussian{S1,S2,S3}) where {T1 <: Real, T2 <: Real, T3 <: Real, S1 <: Real, S2 <: Real, S3 <: Real}
    return PGeneralizedGaussian(T1(d.μ), T2(d.α), T3(d.p), check_args=false)
end

@distr_support PGeneralizedGaussian -Inf Inf


#### Parameters
partype(::PGeneralizedGaussian{T1,T2,T3}) where {T1,T2,T3} = promote_type(T1,T2,T3)

params(d::PGeneralizedGaussian) = (d.μ, d.α, d.p)
location(d::PGeneralizedGaussian) = d.μ
shape(d::PGeneralizedGaussian) = d.p
scale(d::PGeneralizedGaussian) = d.α


#### Statistics

mean(d::PGeneralizedGaussian) = d.μ
median(d::PGeneralizedGaussian) = d.μ
mode(d::PGeneralizedGaussian) = d.μ

var(d::PGeneralizedGaussian) = (d.α^2) * (gamma(3.0 * inv(d.p)) / gamma(inv(d.p)))
std(d::PGeneralizedGaussian) = (d.α) * sqrt(gamma(3.0 * inv(d.p)) / gamma(inv(d.p)))

skewness(d::PGeneralizedGaussian{T1, T2, T3}) where {T1,T2,T3} = zero(T1)
kurtosis(d::PGeneralizedGaussian) = gamma(5.0 * inv(d.p)) * gamma(inv(d.p)) / (gamma(3.0 * inv(d.p))^2) - 3.0
entropy(d::PGeneralizedGaussian) = inv(d.p) - log( d.p / (2.0 * d.α * gamma(inv(d.p))))


#### Evaluation

"""
    pdf(d, x)

Calculates the PDF of the specified distribution 'd'.
"""
function pdf(d::PGeneralizedGaussian, x::Real)
    (μ, α, p) = params(d)
    return ( p / ( 2.0 * α * gamma(1 / p) ) ) * exp( -( abs(x - μ) / α )^p )
end

"""
    cdf(d, x)

Calculates the CDF of the distribution. To determine the CDF, the incomplete
gamma function is required. The CDF  of the Gamma distribution provides this,
with the necessary 1/Γ(a) normalization.
"""
function cdf(d::PGeneralizedGaussian, x::Real)
    (μ, α, p) = params(d)
    v = cdf(Gamma(inv(p), 1), (abs(x - μ) / α)^p) * inv(2)
    return typeof(v)(1/2) + sign(x - μ) * v
end

#### Sampling

"""
    rand(rng, d)

Extract a sample from the p-Generalized Gaussian distribution 'd'. The sampling
procedure is implemented from from [1].
[1]  Gonzalez-Farias, G., Molina, J. A. D., & Rodríguez-Dagnino, R. M. (2009).
Efficiency of the approximated shape parameter estimator in the generalized
Gaussian distribution. IEEE Transactions on Vehicular Technology, 58(8),
4214-4223.
"""
function rand(rng::AbstractRNG, d::PGeneralizedGaussian)

    # utilizing the sampler from the Gamma distribution.
    g = Gamma(inv(d.p), 1)

    # random variable with value -1 or 1 with probability (1/2).
    b = 2.0 * rand(Bernoulli()) -1

    return d.μ + inv(sqrt(d.α)) * rand(rng, g)^inv(d.p) * b
end
