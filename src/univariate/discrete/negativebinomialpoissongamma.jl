"""
Shape-scale parameterization of the negative binomial distribution. For small values of
`β`, this confers superior numerical stability in comparison to the parameterization in terms
`r`, `p` found in `NegativeBinomial`.

```math
P(X = n) = {n + \\alpha - 1 \\choose \\alpha - 1} (\\frac{1}{\\beta + 1})^\\alpha (\\frac{\\beta}{\\beta + 1})^n, \\quad \\text{for } n = 0,1,2,\\ldots
```

In terms of the mixture definition, this corresponds to:
```math
n \\sim \\text{Poisson}(z), \\quad
z \\sim \\text{Gamma}(\\alpha, \\beta)
```

```julia
NegativeBinomialPoissonGamma()        # distribution with α = 1.0 and β = 1.0
NegativeBinomialPoissonGamma(α, β)    # distribution with shape α and scale β

convert(NegativeBinomial{T}, d)                # (r, p) = (α, 1 / (β + 1))
convert(NegativeBinomialLocation{T}, d)        # (μ, ϕ) = (α * β, 1 / α)
convert(NegativeBinomialLogLocation{T}, d)     # (η, ϕ) = (log(α) + log(β), 1 / α)
```

External links:
* [Negative binomial distribution, Bayesian Data Analysis (3rd edition), Appendix A](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)
Note: the definition above uses the scale parameterization of the Gamma distribution
(matching Distributions.jl), whereas BDA uses the inverse scale parameterization.

See also: [`NegativeBinomialLocation`](@ref), [`NegativeBinomialLogLocation`](@ref)
"""
struct NegativeBinomialPoissonGamma{T<:Real} <: DiscreteUnivariateDistribution
    α::T
    β::T
    NegativeBinomialPoissonGamma{T}(α::T, β::T) where {T<:Real} = new{T}(α, β)
end

const PoissonGamma = NegativeBinomialPoissonGamma

function NegativeBinomialPoissonGamma(α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args NegativeBinomialPoissonGamma (α, α > zero(α)) (β, β > zero(β))
    return NegativeBinomialPoissonGamma{T}(α, β)
end

NegativeBinomialPoissonGamma(α::Real, β::Real; check_args::Bool=true) = NegativeBinomialPoissonGamma(promote(α, β)...; check_args=check_args)
NegativeBinomialPoissonGamma(α::Integer, β::Integer; check_args::Bool=true) = NegativeBinomialPoissonGamma(float(α), float(β); check_args=check_args)
NegativeBinomialPoissonGamma(α::Real; check_args::Bool=true) = NegativeBinomialPoissonGamma(α, one(α); check_args=check_args)
NegativeBinomialPoissonGamma() = NegativeBinomialPoissonGamma{Float64}(1.0, 1.0)

@distr_support NegativeBinomialPoissonGamma 0 Inf

insupport(d::NegativeBinomialPoissonGamma, x::Real) = isinteger(x) && x ≥ 0
#### Conversions

function convert(::Type{NegativeBinomialPoissonGamma{T}}, d::NegativeBinomialPoissonGamma) where {T<:Real}
    NegativeBinomialPoissonGamma{T}(T(d.α), T(d.β))
end
convert(::Type{NegativeBinomialPoissonGamma{T}}, d::NegativeBinomialPoissonGamma{T}) where {T<:Real} = d

#### Parameters

params(d::NegativeBinomialPoissonGamma) = (d.α, d.β)
partype(::NegativeBinomialPoissonGamma{T}) where {T} = T

shape(d::NegativeBinomialPoissonGamma) = d.α
scale(d::NegativeBinomialPoissonGamma) = d.β

succprob(d::NegativeBinomialPoissonGamma{T}) where {T} = inv(d.β + one(T))
failprob(d::NegativeBinomialPoissonGamma{T}) where {T} = d.β / (d.β + one(T))

#### Statistics

mean(d::NegativeBinomialPoissonGamma) = d.α * d.β

var(d::NegativeBinomialPoissonGamma{T}) where {T} = d.α * d.β * (one(T) + d.β)

std(d::NegativeBinomialPoissonGamma{T}) where {T} = √(d.α * d.β * (one(T) + d.β))

skewness(d::NegativeBinomialPoissonGamma{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * d.α))

kurtosis(d::NegativeBinomialPoissonGamma{T}) where {T} = (p = succprob(d); T(6) / d.α + (p * p) / ((one(T) - p) * d.α))

mode(d::NegativeBinomialPoissonGamma{T}) where {T} = floor(Int, d.β * (d.α - one(T)))
mode(d::NegativeBinomialPoissonGamma{BigFloat}) = floor(BigInt, d.β * (d.α - one(T)))

#### Evaluation & Sampling

function logpdf(d::NegativeBinomialPoissonGamma, n::Real)
    α, β = params(d)
    r = n * log(β) - (n + α) * log(β + 1)
    if isone(succprob(d)) && iszero(n)
        return zero(r)
    elseif !insupport(d, n)
        return oftype(r, -Inf)
    else
        return r - log(n + α) - logbeta(α, n + 1)
    end
end

rand(rng::AbstractRNG, d::NegativeBinomialPoissonGamma) = rand(rng, Poisson(rand(rng, Gamma(d.α, d.β))))

# cdf and quantile is roundabout, but this is the most reliable approach
cdf(d::NegativeBinomialPoissonGamma{T}, x::Real) where {T} = cdf(convert(NegativeBinomial{T}, d), x)
ccdf(d::NegativeBinomialPoissonGamma{T}, x::Real) where {T} = ccdf(convert(NegativeBinomial{T}, d), x)
logcdf(d::NegativeBinomialPoissonGamma{T}, x::Real) where {T} = logcdf(convert(NegativeBinomial{T}, d), x)
logccdf(d::NegativeBinomialPoissonGamma{T}, x::Real) where {T} = logccdf(convert(NegativeBinomial{T}, d), x)
quantile(d::NegativeBinomialPoissonGamma{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
cquantile(d::NegativeBinomialPoissonGamma{T}, q::Real) where {T} = cquantile(convert(NegativeBinomial{T}, d), q)
invlogcdf(d::NegativeBinomialPoissonGamma{T}, lq::Real) where {T} = invlogcdf(convert(NegativeBinomial{T}, d), lq)
invlogccdf(d::NegativeBinomialPoissonGamma{T}, lq::Real) where {T} = invlogccdf(convert(NegativeBinomial{T}, d), lq)


function mgf(d::NegativeBinomialPoissonGamma{T}, t::Real) where {T}
    α, β = params(d)
    p = inv(β + one(T))
    # ((1 - p) * exp(t))^α / (1 - p * exp(t))^α
    ((one(T) - p) / (inv(exp(t)) - p))^α
end

function cf(d::NegativeBinomialPoissonGamma{T}, t::Real) where {T}
    α, β = params(d)
    p = inv(β + one(T))
    # (((1 - p) * cis(t)) / (1 - p * cis(t)))^α
    ((one(T) - p) / (inv(cis(t)) - p))^α
end
