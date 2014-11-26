immutable BetaPrime <: ContinuousUnivariateDistribution
    betad::Beta

    BetaPrime(α::Float64, β::Float64) = new(Beta(α, β))
    BetaPrime(α::Float64) = new(Beta(α))
    BetaPrime() = new(Beta())
end

@distr_support BetaPrime 0.0 Inf


#### Parameters

params(d::BetaPrime) = params(d.betad)


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

cdf(d::BetaPrime, x::Float64) = cdf(d.betad, x / (1.0 + x))
ccdf(d::BetaPrime, x::Float64) = ccdf(d.betad, x / (1.0 + x))
logcdf(d::BetaPrime, x::Float64) = logcdf(d.betad, x / (1.0 + x))
logccdf(d::BetaPrime, x::Float64) = logccdf(d.betad, x / (1.0 + x))

quantile(d::BetaPrime, p::Float64) = (x = quantile(d.betad, p); x / (1.0 - x))
cquantile(d::BetaPrime, p::Float64) = (x = cquantile(d.betad, p); x / (1.0 - x))
invlogcdf(d::BetaPrime, p::Float64) = (x = invlogcdf(d.betad, p); x / (1.0 - x))
invlogccdf(d::BetaPrime, p::Float64) = (x = invlogccdf(d.betad, p); x / (1.0 - x))
    

#### Sampling

function rand(d::BetaPrime) 
    (α, β) = params(d)
    rand(Gamma(α)) / rand(Gamma(β))
end

