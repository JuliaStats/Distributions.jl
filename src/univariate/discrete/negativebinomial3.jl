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
NegativeBinomial3()        # distribution with α = 1.0 and β = 1.0
NegativeBinomial3(α, β)    # distribution with shape α and scale β

convert(NegativeBinomial{T}, d)         # Parametric conversion to NegativeBinomial
convert(NegativeBinomial2Log{T}, d)     # Parametric conversion to NegativeBinomial2Log
convert(NegativeBinomial3{T}, d)        # Parametric conversion to NegativeBinomial3
```

External links:
* [Negative binomial distribution, Bayesian Data Analysis (3rd edition), Appendix A](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)
Note: the definition above uses the scale parameterization of the Gamma distribution
(matching Distributions.jl), whereas BDA uses the inverse scale parameterization.

See also: [`NegativeBinomial2`](@ref), [`NegativeBinomial2Log`](@ref)
"""
struct NegativeBinomial3{T<:Real} <: DiscreteUnivariateDistribution
    α::T
    β::T
    NegativeBinomial3{T}(α::T, β::T) where {T<:Real} = new{T}(α, β)
end

function NegativeBinomial3(α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args NegativeBinomial3 (α, α > zero(α)) (β, β > zero(β))
    return NegativeBinomial3{T}(α, β)
end

NegativeBinomial3(α::Real, β::Real; check_args::Bool=true) = NegativeBinomial3(promote(α, β)...; check_args=check_args)
NegativeBinomial3(α::Integer, β::Integer; check_args::Bool=true) = NegativeBinomial3(float(α), float(β); check_args=check_args)
NegativeBinomial3(α::Real; check_args::Bool=true) = NegativeBinomial3(α, 1.0; check_args=check_args)
NegativeBinomial3() = NegativeBinomial3{Float64}(1.0, 1.0)

@distr_support NegativeBinomial3 0 Inf

insupport(d::NegativeBinomial3, x::Real) = false
insupport(d::NegativeBinomial3, x::T) where {T<:Integer} = x ≥ 0
#### Conversions

function convert(::Type{NegativeBinomial3{T}}, α::Real, β::Real) where {T<:Real}
    NegativeBinomial3(T(α), T(β))
end
function convert(::Type{NegativeBinomial3{T}}, d::NegativeBinomial3) where {T<:Real}
    NegativeBinomial3{T}(T(d.α), T(d.β))
end
convert(::Type{NegativeBinomial3{T}}, d::NegativeBinomial3{T}) where {T<:Real} = d

#### Parameters

params(d::NegativeBinomial3) = (d.α, d.β)
partype(::NegativeBinomial3{T}) where {T} = T

succprob(d::NegativeBinomial3{T}) where {T} = inv(d.β + one(T))
failprob(d::NegativeBinomial3{T}) where {T} = (β = d.β; β / (β + one(T)))

#### Statistics

mean(d::NegativeBinomial3{T}) where {T} = d.α * d.β

var(d::NegativeBinomial3{T}) where {T} = d.α * d.β * (one(T) + d.β)

std(d::NegativeBinomial3{T}) where {T} = √(d.α * d.β * (one(T) + d.β))

skewness(d::NegativeBinomial3{T}) where {T} = (p = succprob(d); (T(2) - p) / sqrt((one(T) - p) * d.α))

kurtosis(d::NegativeBinomial3{T}) where {T} = (p = succprob(d); T(6) / d.α + (p * p) / ((one(T) - p) * d.α))

mode(d::NegativeBinomial3{T}) where {T} = d.α > one(T) ? floor(Int, d.β * (d.α - one(T))) : 0

#### Evaluation & Sampling

function logpdf(d::NegativeBinomial3, n::Real)
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

rand(rng::AbstractRNG, d::NegativeBinomial3) = rand(rng, Poisson(rand(rng, Gamma(d.α, d.β))))

# cdf and quantile is roundabout, but this is the most reliable approach
cdf(d::NegativeBinomial3{T}, x::Real) where {T} = cdf(convert(NegativeBinomial{T}, d), x)
ccdf(d::NegativeBinomial3{T}, x::Real) where {T} = ccdf(convert(NegativeBinomial{T}, d), x)
logcdf(d::NegativeBinomial3{T}, x::Real) where {T} = logcdf(convert(NegativeBinomial{T}, d), x)
logccdf(d::NegativeBinomial3{T}, x::Real) where {T} = logccdf(convert(NegativeBinomial{T}, d), x)
quantile(d::NegativeBinomial3{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
cquantile(d::NegativeBinomial3{T}, q::Real) where {T} = quantile(convert(NegativeBinomial{T}, d), q)
invlogcdf(d::NegativeBinomial3{T}, lq::Real) where {T} = invlogcdf(convert(NegativeBinomial{T}, d), lq)
invlogccdf(d::NegativeBinomial3{T}, lq::Real) where {T} = invlogccdf(convert(NegativeBinomial{T}, d), lq)


function mgf(d::NegativeBinomial3, t::Real)
    α, β = params(d)
    p = inv(β + 1)
    # ((1 - p) * exp(t))^α / (1 - p * exp(t))^α
    ((1 - p) / (inv(exp(t)) - p))^α
end

function cf(d::NegativeBinomial3, t::Real)
    α, β = params(d)
    p = inv(β + 1)
    # (((1 - p) * cis(t)) / (1 - p * cis(t)))^α
    ((1 - p) / (inv(cis(t)) - p))^α
end
