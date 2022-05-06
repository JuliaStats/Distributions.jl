"""
Alternative parameterization of the negative binomial distribution in terms of
location `μ` and overdispersion `ϕ`.

```math
P(X = n) = {n + \\phi^-1 - 1 \\choose n} (\\frac{\\mu}{\\mu + \\phi^-1})^n (\\frac{\\phi^-1}{\\mu + \\phi^-1})^\\phi^-1, \\quad \\text{for } n = 0,1,2,\\ldots
```

In terms of the mixture definition, this corresponds to:
```math
n \\sim \\text{Poisson}(zμ), \\quad
z \\sim \\text{Gamma}(\\phi^-1, \\phi)
```
This provides a clear interpretation of the negative binomial as an overdispersed
Poisson with expected location 𝔼[z] = μ and var[z] = μ + μ²ϕ. It also has advantages in that
its pmf can be expressed in terms of μ = exp(η), such that it can natively be
parameterized on the log scale.

```julia
NegativeBinomialLocation()        # distribution with μ = 1.0 and ϕ = 1.0
NegativeBinomialLocation(μ, ϕ)    # distribution with location μ and overdispersion ϕ

convert(NegativeBinomial{T}, d)                    # (r, p) = (1 / ϕ, 1 / (ϕ * μ + 1))
convert(NegativeBinomialLogLocation{T}, d)         # (η, ϕ) = (log(μ), ϕ)
convert(NegativeBinomialPoissonGamma{T}, d)        # (α, β) = (1 / ϕ, ϕ * μ)
```

External links:
* [Negative binomial distribution (alternative paramerization), Stan](https://mc-stan.org/docs/2_29/functions-reference/nbalt.html)
Note: the definition above uses the scale parameterization of the Gamma distribution
(matching Distributions.jl), whereas Stan uses the inverse scale parameterization.

See also: [`NegativeBinomialLogLocation`](@ref), [`NegativeBinomialPoissonGamma`](@ref)
"""
struct NegativeBinomialLocation{T<:Real} <: DiscreteUnivariateDistribution
    μ::T
    ϕ::T
    NegativeBinomialLocation{T}(μ::T, ϕ::T) where {T<:Real} = new{T}(μ, ϕ)
end

function NegativeBinomialLocation(μ::T, ϕ::T; check_args::Bool=true) where {T<:Real}
    @check_args NegativeBinomialLocation (μ, μ > zero(μ)) (ϕ, ϕ ≥ zero(ϕ))
    return NegativeBinomialLocation{T}(μ, ϕ)
end

NegativeBinomialLocation(μ::Real, ϕ::Real; check_args::Bool=true) = NegativeBinomialLocation(promote(μ, ϕ)...; check_args=check_args)
NegativeBinomialLocation(μ::Integer, ϕ::Integer; check_args::Bool=true) = NegativeBinomialLocation(float(μ), float(ϕ); check_args=check_args)
NegativeBinomialLocation(μ::Real; check_args::Bool=true) = NegativeBinomialLocation(μ, one(μ); check_args=check_args)
NegativeBinomialLocation() = NegativeBinomialLocation{Float64}(1.0, 1.0)

@distr_support NegativeBinomialLocation 0 Inf

insupport(d::NegativeBinomialLocation, x::Real) = isinteger(x) && x ≥ 0
#### Conversions

function convert(::Type{NegativeBinomialLocation{T}}, d::NegativeBinomialLocation) where {T<:Real}
    NegativeBinomialLocation{T}(T(d.μ), T(d.ϕ))
end
convert(::Type{NegativeBinomialLocation{T}}, d::NegativeBinomialLocation{T}) where {T<:Real} = d

#### Parameters

params(d::NegativeBinomialLocation) = (d.μ, d.ϕ)
partype(::NegativeBinomialLocation{T}) where {T} = T

succprob(d::NegativeBinomialLocation{T}) where {T} = inv(d.μ * d.ϕ + one(T))
failprob(d::NegativeBinomialLocation{T}) where {T} = inv(one(T) + inv(d.μ * d.ϕ))

#### Statistics

mean(d::NegativeBinomialLocation) = d.μ

var(d::NegativeBinomialLocation{T}) where {T} = d.μ * (one(T) + d.μ * d.ϕ)

std(d::NegativeBinomialLocation{T}) where {T} = √(d.μ * (one(T) + d.μ * d.ϕ))

skewness(d::NegativeBinomialLocation{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * inv(d.ϕ)))

kurtosis(d::NegativeBinomialLocation{T}) where {T} = (p = succprob(d); T(6) * d.ϕ + p * p * d.ϕ / (one(T) - p))

mode(d::NegativeBinomialLocation) = floor(Int, d.μ - d.μ * d.ϕ)
mode(d::NegativeBinomialLocation{BigFloat}) = floor(BigInt, d.μ - d.μ * d.ϕ)

#### Evaluation & Sampling

function logpdf(d::NegativeBinomialLocation, n::Real)
    μ, ϕ = params(d)
    r = -n * log1p(inv(μ * ϕ)) - log1p(μ * ϕ) / ϕ
    if isone(succprob(d)) && iszero(n)
        return zero(r)
    elseif !insupport(d, n)
        return oftype(r, -Inf)
    else
        return r - log(n + inv(ϕ)) - logbeta(inv(ϕ), n + 1)
    end
end

rand(rng::AbstractRNG, d::NegativeBinomialLocation) = rand(rng, Poisson(d.μ * rand(rng, Gamma(inv(d.ϕ), d.ϕ))))

# cdf and quantile is roundabout, but this is the most reliable approach
cdf(d::NegativeBinomialLocation{T}, x::Real) where {T} = cdf(convert(NegativeBinomial{T}, d), x)
ccdf(d::NegativeBinomialLocation{T}, x::Real) where {T} = ccdf(convert(NegativeBinomial{T}, d), x)
logcdf(d::NegativeBinomialLocation{T}, x::Real) where {T} = logcdf(convert(NegativeBinomial{T}, d), x)
logccdf(d::NegativeBinomialLocation{T}, x::Real) where {T} = logccdf(convert(NegativeBinomial{T}, d), x)
quantile(d::NegativeBinomialLocation{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
cquantile(d::NegativeBinomialLocation{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
invlogcdf(d::NegativeBinomialLocation{T}, lq::Real) where {T} = invlogcdf(convert(NegativeBinomial{T}, d), lq)
invlogccdf(d::NegativeBinomialLocation{T}, lq::Real) where {T} = invlogccdf(convert(NegativeBinomial{T}, d), lq)


function mgf(d::NegativeBinomialLocation{T}, t::Real) where {T}
    μ, ϕ = params(d)
    p = inv(μ * ϕ + one(T))
    ((one(T) - p) / (inv(exp(t)) - p))^inv(ϕ)
end

function cf(d::NegativeBinomialLocation{T}, t::Real) where {T}
    μ, ϕ = params(d)
    p = inv(μ * ϕ + one(T))
    ((one(T) - p) / (inv(cis(t)) - p))^inv(ϕ)
end
