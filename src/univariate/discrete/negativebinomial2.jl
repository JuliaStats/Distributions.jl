"""
Alternative location-scale parameterization of the negative binomial distribution
in terms of location `μ` and overdispersion `ϕ`.

```math
P(X = n) = {n + \\phi - 1 \\choose n} (\\frac{\\mu}{\\mu + \\phi})^n (\\frac{\\phi}{\\mu + \\phi})^\\phi, \\quad \\text{for } n = 0,1,2,\\ldots
```

In terms of the mixture definition, this corresponds to:
```math
n \\sim \\text{Poisson}(z), \\quad
z \\sim \\text{Gamma}(\\phi, \\frac{\\mu}{\\phi})
```
This provides a clear interpretation of the negative binomial as an overdispersed
Poisson with expected location 𝔼[z] = μ and var[z] = μ + μ²/ϕ. It also has advantages in that
its pmf can be expressed in terms of μ = exp(η), such that it can natively be
parameterized on the log scale.

```julia
NegativeBinomial2()        # distribution with μ = 1.0 and ϕ = 1.0
NegativeBinomial2(μ, ϕ)    # distribution with location μ and overdispersion ϕ

convert(NegativeBinomial{T}, d)         # Parametric conversion to NegativeBinomial
convert(NegativeBinomial2Log{T}, d)     # Parametric conversion to NegativeBinomial2Log
convert(NegativeBinomial3{T}, d)        # Parametric conversion to NegativeBinomial3
```

External links:
* [Negative binomial distribution (alternative paramerization), Stan](https://mc-stan.org/docs/2_29/functions-reference/nbalt.html)
Note: the definition above uses the scale parameterization of the Gamma distribution
(matching Distributions.jl), whereas Stan uses the inverse scale parameterization.

See also: [`NegativeBinomial2Log`](@ref), [`NegativeBinomial3`](@ref)
"""
struct NegativeBinomial2{T<:Real} <: DiscreteUnivariateDistribution
    μ::T
    ϕ::T
    NegativeBinomial2{T}(μ::T, ϕ::T) where {T<:Real} = new{T}(μ, ϕ)
end

function NegativeBinomial2(μ::T, ϕ::T; check_args::Bool=true) where {T<:Real}
    @check_args NegativeBinomial2 (μ, μ > zero(μ)) (ϕ, ϕ > zero(ϕ))
    return NegativeBinomial2{T}(μ, ϕ)
end

NegativeBinomial2(μ::Real, ϕ::Real; check_args::Bool=true) = NegativeBinomial2(promote(μ, ϕ)...; check_args=check_args)
NegativeBinomial2(μ::Integer, ϕ::Integer; check_args::Bool=true) = NegativeBinomial2(float(μ), float(ϕ); check_args=check_args)
NegativeBinomial2(μ::Real; check_args::Bool=true) = NegativeBinomial2(μ, 1.0; check_args=check_args)
NegativeBinomial2() = NegativeBinomial2{Float64}(1.0, 1.0)

@distr_support NegativeBinomial2 0 Inf

insupport(d::NegativeBinomial2, x::Real) = false
insupport(d::NegativeBinomial2, x::T) where {T<:Integer} = x ≥ 0
#### Conversions

function convert(::Type{NegativeBinomial2{T}}, μ::Real, ϕ::Real) where {T<:Real}
    NegativeBinomial2(T(μ), T(ϕ))
end
function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial2) where {T<:Real}
    NegativeBinomial2{T}(T(d.μ), T(d.ϕ))
end
convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial2{T}) where {T<:Real} = d

# # Interconversion
# function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial) where {T<:Real}
#     NegativeBinomial2{T}(T(d.r * (1 - d.p) / d.p), T(d.r))
# end
# function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial2Log) where {T<:Real}
#     NegativeBinomial2{T}(T(exp(d.η)), T(d.ϕ))
# end
# function convert(::Type{NegativeBinomial2{T}}, d::NegativeBinomial3) where {T<:Real}
#     NegativeBinomial2{T}(T(d.α * d.β), T(d.α))
# end

#### Parameters

params(d::NegativeBinomial2) = (d.μ, d.ϕ)
partype(::NegativeBinomial2{T}) where {T} = T

succprob(d::NegativeBinomial2{T}) where {T} = d.ϕ / (d.μ + d.ϕ)
failprob(d::NegativeBinomial2{T}) where {T} = d.μ / (d.μ + d.ϕ)

#### Statistics

mean(d::NegativeBinomial2{T}) where {T} = d.μ

var(d::NegativeBinomial2{T}) where {T} = (μ, ϕ = params(d); μ * (one(T) + μ / ϕ))

std(d::NegativeBinomial2{T}) where {T} = (μ, ϕ = params(d); √(μ * (one(T) + μ / ϕ)))

skewness(d::NegativeBinomial2{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * d.ϕ))

kurtosis(d::NegativeBinomial2{T}) where {T} = (p = succprob(d); T(6) / d.ϕ + (p * p) / ((one(T) - p) * d.ϕ))

mode(d::NegativeBinomial2{T}) where {T} = (μ, ϕ = params(d); ϕ > one(T) ? floor(Int, μ * (ϕ - one(T)) / ϕ) : 0)

#### Evaluation & Sampling

function logpdf(d::NegativeBinomial2, n::Real)
    μ, ϕ = params(d)
    r = n * log(μ) + ϕ * log(ϕ) - (n + ϕ) * log(μ + ϕ)
    if isone(succprob(d)) && iszero(n)
        return zero(r)
    elseif !insupport(d, n)
        return oftype(r, -Inf)
    else
        return r - log(n + ϕ) - logbeta(ϕ, n + 1)
    end
end

rand(rng::AbstractRNG, d::NegativeBinomial2) = ((; μ, ϕ) = d; rand(rng, Poisson(rand(rng, Gamma(ϕ, μ / ϕ)))))

# cdf and quantile is roundabout, but this is the most reliable approach
cdf(d::NegativeBinomial2{T}, x::Real) where {T} = cdf(convert(NegativeBinomial{T}, d), x)
ccdf(d::NegativeBinomial2{T}, x::Real) where {T} = ccdf(convert(NegativeBinomial{T}, d), x)
logcdf(d::NegativeBinomial2{T}, x::Real) where {T} = logcdf(convert(NegativeBinomial{T}, d), x)
logccdf(d::NegativeBinomial2{T}, x::Real) where {T} = logccdf(convert(NegativeBinomial{T}, d), x)
quantile(d::NegativeBinomial2{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
cquantile(d::NegativeBinomial2{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
invlogcdf(d::NegativeBinomial2{T}, lq::Real) where {T} = invlogcdf(convert(NegativeBinomial{T}, d), lq)
invlogccdf(d::NegativeBinomial2{T}, lq::Real) where {T} = invlogccdf(convert(NegativeBinomial{T}, d), lq)


function mgf(d::NegativeBinomial2, t::Real)
    μ, ϕ = params(d)
    p = ϕ / (μ + ϕ)
    # ((1 - p) * exp(t))^ϕ / (1 - p * exp(t))^ϕ
    ((1 - p) / (inv(exp(t)) - p))^ϕ
end

function cf(d::NegativeBinomial2, t::Real)
    μ, ϕ = params(d)
    p = ϕ / (μ + ϕ)
    # (((1 - p) * cis(t)) / (1 - p * cis(t)))^ϕ
    ((1 - p) / (inv(cis(t)) - p))^ϕ
end
