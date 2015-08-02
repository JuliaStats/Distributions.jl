immutable BetaPrime <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64

    function BetaPrime(α::Float64, β::Float64)
        (α > zero(α) && β > zero(β)) || error("α and β must be positive")
        @compat new(Float64(α), Float64(β))
    end

    BetaPrime(α::Float64) = BetaPrime(α)
    BetaPrime() = new(1.0, 1.0)
end

@distr_support BetaPrime 0.0 Inf

#### Parameters

params(d::BetaPrime) = (d.α, d.β)


#### Statistics

mean(d::BetaPrime) = ((α, β) = params(d); β > 1.0 ? α / (β - 1.0) : NaN)

mode(d::BetaPrime) = ((α, β) = params(d); α > 1.0 ? (α - 1.0) / (β + 1.0) : 0.0)

function var(d::BetaPrime)
    (α, β) = params(d)
    β > 2.0 ? α * (α + β - 1.0) / ((β - 2.0) * (β - 1.0)^2) : NaN
end

function skewness(d::BetaPrime)
    (α, β) = params(d)
    if β > 3.0
        s = α + β - 1.0
        2.0 * (α + s) / (β - 3.0) * sqrt((β - 2.0) / (α * s))
    else
        return NaN
    end
end


#### Evaluation

function logpdf(d::BetaPrime, x::Float64)
    (α, β) = params(d)
    (α - 1.0) * log(x) - (α + β) * log1p(x) - lbeta(α, β)
end

pdf(d::BetaPrime, x::Float64) = exp(logpdf(d, x))

cdf(d::BetaPrime, x::Float64) = betacdf(d.α, d.β, x / (1.0 + x))
ccdf(d::BetaPrime, x::Float64) = betaccdf(d.α, d.β, x / (1.0 + x))
logcdf(d::BetaPrime, x::Float64) = betalogcdf(d.α, d.β, x / (1.0 + x))
logccdf(d::BetaPrime, x::Float64) = betalogccdf(d.α, d.β, x / (1.0 + x))

quantile(d::BetaPrime, p::Float64) = (x = betainvcdf(d.α, d.β, p); x / (1.0 - x))
cquantile(d::BetaPrime, p::Float64) = (x = betainvccdf(d.α, d.β, p); x / (1.0 - x))
invlogcdf(d::BetaPrime, p::Float64) = (x = betainvlogcdf(d.α, d.β, p); x / (1.0 - x))
invlogccdf(d::BetaPrime, p::Float64) = (x = betainvlogccdf(d.α, d.β, p); x / (1.0 - x))


#### Sampling

function rand(d::BetaPrime)
    (α, β) = params(d)
    rand(Gamma(α)) / rand(Gamma(β))
end
