"""
Alternative parameterization of the negative binomial distribution in terms of
the log location `η` and overdispersion `ϕ`, such that

```math
\\text{NegativeBinomialLogLocation}(n | \\eta, \\phi) = \\text{NegativeBinomialLocation}(n | \\exp(\\eta), \\phi)
```

This paramerization is important in that it connects the the negative binomial distribution
to the classic form of a Poisson GLM with offset (with logarithmic link),
the rate parameter of which is expressed as λ = μT, with T an exposure and μ being events
per unit exposure. Moreover, paramerization on the log scale enables evaluation of
the `logpdf` as:

`log(binomial(n + ϕ⁻¹ - 1, n)) + n * η - n * log(exp(η) * ϕ + 1)  + n * log(ϕ) - log(exp(η) * ϕ + 1) / ϕ`

(note: equation for clarity, not for implementation).
This results in fewer special function calls for each `logpdf` evaluation compared
to the NegativeBinomialLocation(μ, ϕ).

```julia
NegativeBinomialLogLocation()        # distribution with η = 0.0 and ϕ = 1.0
NegativeBinomialLogLocation(η, ϕ)    # distribution with log location η and overdispersion ϕ

convert(NegativeBinomial{T}, d)                 # (r, p) = (1 / ϕ, 1 / (ϕ * exp(η) + 1))
convert(NegativeBinomialLocation{T}, d)         # (μ, ϕ) = (exp(η), ϕ)
convert(NegativeBinomialPoissonGamma{T}, d)     # (α, β) = (1 / ϕ, ϕ * exp(η))
```

External links:
* [Negative binomial distribution (log alternative parameterization), Stan](https://mc-stan.org/docs/2_29/functions-reference/neg-binom-2-log.html)

See also: [`NegativeBinomialLocation`](@ref), [`NegativeBinomialPoissonGamma`](@ref)
"""
struct NegativeBinomialLogLocation{T<:Real} <: DiscreteUnivariateDistribution
    η::T
    ϕ::T
    NegativeBinomialLogLocation{T}(η::T, ϕ::T) where {T<:Real} = new{T}(η, ϕ)
end

function NegativeBinomialLogLocation(η::T, ϕ::T; check_args::Bool=true) where {T<:Real}
    @check_args NegativeBinomialLogLocation (η, η > zero(η)) (ϕ, ϕ ≥ zero(ϕ))
    return NegativeBinomialLogLocation{T}(η, ϕ)
end

NegativeBinomialLogLocation(η::Real, ϕ::Real; check_args::Bool=true) = NegativeBinomialLogLocation(promote(η, ϕ)...; check_args=check_args)
NegativeBinomialLogLocation(η::Integer, ϕ::Integer; check_args::Bool=true) = NegativeBinomialLogLocation(float(η), float(ϕ); check_args=check_args)
NegativeBinomialLogLocation(η::Real; check_args::Bool=true) = NegativeBinomialLogLocation(η, one(η); check_args=check_args)
NegativeBinomialLogLocation() = NegativeBinomialLogLocation{Float64}(0.0, 1.0)

@distr_support NegativeBinomialLogLocation 0 Inf

insupport(d::NegativeBinomialLogLocation, x::Real) = isinteger(x) && x ≥ 0
#### Conversions

function convert(::Type{NegativeBinomialLogLocation{T}}, d::NegativeBinomialLogLocation) where {T<:Real}
    NegativeBinomialLogLocation{T}(T(d.η), T(d.ϕ))
end
convert(::Type{NegativeBinomialLogLocation{T}}, d::NegativeBinomialLogLocation{T}) where {T<:Real} = d

#### Parameters

params(d::NegativeBinomialLogLocation) = (d.η, d.ϕ)
partype(::NegativeBinomialLogLocation{T}) where {T} = T

succprob(d::NegativeBinomialLogLocation{T}) where {T} = inv(exp(d.η) * d.ϕ + one(T))
failprob(d::NegativeBinomialLogLocation{T}) where {T} = inv(one(T) + inv(exp(d.η) * d.ϕ))

#### Statistics

mean(d::NegativeBinomialLogLocation) = exp(d.η)

var(d::NegativeBinomialLogLocation{T}) where {T} = (μ = exp(d.η); μ * (one(T) + μ * d.ϕ))

std(d::NegativeBinomialLogLocation{T}) where {T} = (μ = exp(d.η); √(μ * (one(T) + μ * d.ϕ)))

skewness(d::NegativeBinomialLogLocation{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * inv(d.ϕ)))

kurtosis(d::NegativeBinomialLogLocation{T}) where {T} = (p = succprob(d); T(6) * d.ϕ + p * p * d.ϕ / (one(T) - p))

mode(d::NegativeBinomialLogLocation) = (μ = exp(d.η); floor(Int, μ - μ * d.ϕ))
mode(d::NegativeBinomialLogLocation{BigFloat}) = (μ = exp(d.η); floor(BigInt, μ - μ * d.ϕ))

#### Evaluation & Sampling

function logpdf(d::NegativeBinomialLogLocation, n::Real)
    η, ϕ = params(d)
    r = log1p(exp(η) * ϕ)
    if isone(succprob(d)) && iszero(n)
        return zero(r)
    elseif !insupport(d, n)
        return oftype(r, -Inf)
    else
        ϕ⁻¹ = inv(ϕ)
        c = loggamma(n + ϕ⁻¹) - loggamma(n + 1) - loggamma(ϕ⁻¹)
        return c + n * η - n * r + n * log(ϕ) - r / ϕ
    end
end

rand(rng::AbstractRNG, d::NegativeBinomialLogLocation) = rand(rng, Poisson(exp(d.η) * rand(rng, Gamma(inv(d.ϕ), d.ϕ))))

# cdf and quantile is roundabout, but this is the most reliable approach
cdf(d::NegativeBinomialLogLocation{T}, x::Real) where {T} = cdf(convert(NegativeBinomial{T}, d), x)
ccdf(d::NegativeBinomialLogLocation{T}, x::Real) where {T} = ccdf(convert(NegativeBinomial{T}, d), x)
logcdf(d::NegativeBinomialLogLocation{T}, x::Real) where {T} = logcdf(convert(NegativeBinomial{T}, d), x)
logccdf(d::NegativeBinomialLogLocation{T}, x::Real) where {T} = logccdf(convert(NegativeBinomial{T}, d), x)
quantile(d::NegativeBinomialLogLocation{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
cquantile(d::NegativeBinomialLogLocation{T}, q::Real) where {T} = cquantile(convert(NegativeBinomial{T}, d), q)
invlogcdf(d::NegativeBinomialLogLocation{T}, lq::Real) where {T} = invlogcdf(convert(NegativeBinomial{T}, d), lq)
invlogccdf(d::NegativeBinomialLogLocation{T}, lq::Real) where {T} = invlogccdf(convert(NegativeBinomial{T}, d), lq)


function mgf(d::NegativeBinomialLogLocation{T}, t::Real) where {T}
    η, ϕ = params(d)
    p = inv(exp(η) * ϕ + one(T))
    ((one(T) - p) / (inv(exp(t)) - p))^inv(ϕ)
end

function cf(d::NegativeBinomialLogLocation{T}, t::Real) where {T}
    η, ϕ = params(d)
    p = inv(exp(η) * ϕ + one(T))
    ((one(T) - p) / (inv(cis(t)) - p))^inv(ϕ)
end
