"""
Related to the location-scale parameterization of `NegativeBinomial2`, the following
parameterization is defined in terms of the log location parameter, `η`,
such that `NegativeBinomial2Log(η, ϕ)` = `NegativeBinomial2(exp(η), ϕ)`.

This paramerization is important in that it connects the the negative binomial distribution
to the classic form of a Poisson GLM with offset (with logarithmic link),
the rate parameter of which is expressed as λ = μT, with T an exposure and μ being events
per unit exposure. Moreover, paramerization on the log scale enables evaluation of
the `logpdf` as:

`log(binomial(n + ϕ - 1, n)) + n * η - n * (log(exp(η) / ϕ + 1) + log(ϕ)) - ϕ * log(exp(η) / ϕ + 1)`

(note: equation for clarity, not for implementation).
This results in fewer special function calls for each `logpdf` evaluation compared
to the NegativeBinomial(μ, ϕ).

```julia
NegativeBinomial2Log()        # distribution with η = 0.0 and ϕ = 1.0
NegativeBinomial2Log(η, ϕ)    # distribution with log location η and overdispersion ϕ

convert(NegativeBinomial{T}, d)      # Parametric conversion to NegativeBinomial
convert(NegativeBinomial2{T}, d)     # Parametric conversion to NegativeBinomial2
convert(NegativeBinomial3{T}, d)     # Parametric conversion to NegativeBinomial3
```

External links:
* [Negative binomial distribution (log alternative parameterization), Stan](https://mc-stan.org/docs/2_29/functions-reference/neg-binom-2-log.html)

See also: [`NegativeBinomial2`](@ref), [`NegativeBinomial3`](@ref)
"""
struct NegativeBinomial2Log{T<:Real} <: DiscreteUnivariateDistribution
    η::T
    ϕ::T
    NegativeBinomial2Log{T}(η::T, ϕ::T) where {T<:Real} = new{T}(η, ϕ)
end

function NegativeBinomial2Log(η::T, ϕ::T; check_args::Bool=true) where {T<:Real}
    @check_args NegativeBinomial2Log (η, η > -Inf) (ϕ, ϕ > zero(ϕ))
    return NegativeBinomial2Log{T}(η, ϕ)
end

NegativeBinomial2Log(η::Real, ϕ::Real; check_args::Bool=true) = NegativeBinomial2Log(promote(η, ϕ)...; check_args=check_args)
NegativeBinomial2Log(η::Real; check_args::Bool=true) = NegativeBinomial2Log(η, one(η); check_args=check_args)
NegativeBinomial2Log() = NegativeBinomial2Log{Float64}(0.0, 1.0)

@distr_support NegativeBinomial2Log 0 Inf

insupport(d::NegativeBinomial2Log, x::Real) = false
insupport(d::NegativeBinomial2Log, x::T) where {T<:Integer} = x ≥ 0
#### Conversions

function convert(::Type{NegativeBinomial2Log{T}}, η::Real, ϕ::Real) where {T<:Real}
    NegativeBinomial2Log(T(η), T(ϕ))
end
function convert(::Type{NegativeBinomial2Log{T}}, d::NegativeBinomial2Log) where {T<:Real}
    NegativeBinomial2Log{T}(T(d.η), T(d.ϕ))
end
convert(::Type{NegativeBinomial2Log{T}}, d::NegativeBinomial2Log{T}) where {T<:Real} = d


#### Parameters

params(d::NegativeBinomial2Log) = (d.η, d.ϕ)
partype(::NegativeBinomial2Log{T}) where {T} = T

succprob(d::NegativeBinomial2Log{T}) where {T} = d.ϕ / (exp(d.η) + d.ϕ)
failprob(d::NegativeBinomial2Log{T}) where {T} = (μ = exp(d.η); μ / (μ + d.ϕ))

#### Statistics

mean(d::NegativeBinomial2Log{T}) where {T} = d.η

var(d::NegativeBinomial2Log{T}) where {T} = (μ = exp(d.η); μ * (one(T) + μ / d.ϕ))

std(d::NegativeBinomial2Log{T}) where {T} = (μ = exp(d.η); √(μ * (one(T) + μ / d.ϕ)))

skewness(d::NegativeBinomial2Log{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * d.ϕ))

kurtosis(d::NegativeBinomial2Log{T}) where {T} = (p = succprob(d); T(6) / d.ϕ + (p * p) / ((one(T) - p) * d.ϕ))

mode(d::NegativeBinomial2Log{T}) where {T} = d.ϕ > one(T) ? floor(Int, exp(d.η) * (d.ϕ - one(T)) / d.ϕ) : 0

#### Evaluation & Sampling
@inline binomial_log(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

function logpdf(d::NegativeBinomial2Log, n::Real)
    η, ϕ = params(d)
    # ϕₘ₁ = ϕ - 1
    # c = log(n + ϕₘ₁) - log(n * ϕₘ₁) - logbeta(n, ϕₘ₁)
    # c = binomial_log(n + ϕₘ₁, n) # safer; TO DO: to create branches to use logbeta
    r = log1p(exp(η) / ϕ)
    if isone(succprob(d)) && iszero(n)
        return zero(r)
    elseif !insupport(d, n)
        return oftype(r, -Inf)
    else
        c = loggamma(n + ϕ) - loggamma(n + 1) - loggamma(ϕ)
        return c + n * η - n * (r + log(ϕ)) - ϕ * r
    end
end

rand(rng::AbstractRNG, d::NegativeBinomial2Log) = rand(rng, Poisson(rand(rng, Gamma(d.ϕ, exp(d.η) / d.ϕ))))

# cdf and quantile is roundabout, but this is the most reliable approach
cdf(d::NegativeBinomial2Log{T}, x::Real) where {T} = cdf(convert(NegativeBinomial{T}, d), x)
ccdf(d::NegativeBinomial2Log{T}, x::Real) where {T} = ccdf(convert(NegativeBinomial{T}, d), x)
logcdf(d::NegativeBinomial2Log{T}, x::Real) where {T} = logcdf(convert(NegativeBinomial{T}, d), x)
logccdf(d::NegativeBinomial2Log{T}, x::Real) where {T} = logccdf(convert(NegativeBinomial{T}, d), x)
quantile(d::NegativeBinomial2Log{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
cquantile(d::NegativeBinomial2Log{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
invlogcdf(d::NegativeBinomial2Log{T}, lq::Real) where {T} = invlogcdf(convert(NegativeBinomial{T}, d), lq)
invlogccdf(d::NegativeBinomial2Log{T}, lq::Real) where {T} = invlogccdf(convert(NegativeBinomial{T}, d), lq)



function mgf(d::NegativeBinomial2Log, t::Real)
    η, ϕ = params(d)
    p = ϕ / (exp(η) + ϕ)
    # ((1 - p) * exp(t))^ϕ / (1 - p * exp(t))^ϕ
    ((1 - p) / (inv(exp(t)) - p))^ϕ
end

function cf(d::NegativeBinomial2Log, t::Real)
    η, ϕ = params(d)
    p = ϕ / (exp(η) + ϕ)
    # (((1 - p) * cis(t)) / (1 - p * cis(t)))^ϕ
    ((1 - p) / (inv(cis(t)) - p))^ϕ
end
