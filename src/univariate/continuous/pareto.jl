immutable Pareto <: ContinuousUnivariateDistribution
    α::Float64
    s::Float64

    function Pareto(α::Real, s::Real)
        (α > zero(α) && s > zero(s)) || error("Pareto: shape and scale must be positive")
        new(float64(α), float64(s))
    end

    Pareto(α::Real) = Pareto(α, 1.0)
    Pareto() = new(1.0, 1.0)
end

@distr_support Pareto d.s Inf


#### Parameters

shape(d::Pareto) = d.α
scale(d::Pareto) = d.s

params(d::Pareto) = (d.α, d.s)


#### Statistics

mean(d::Pareto) = ((α, s) = params(d); α > 1.0 ? α * s / (α - 1.0) : Inf)
median(d::Pareto) = ((α, s) = params(d); s * 2.0 ^ (1.0 / α))
mode(d::Pareto) = d.s

function var(d::Pareto)
    (α, s) = params(d)
    α > 2.0 ? (s^2 * α) / ((α - 1.0)^2 * (α - 2.0)) : Inf
end

function skewness(d::Pareto)
    α = shape(d)
    α > 3.0 ? ((2.0 * (1.0 + α)) / (α - 3.0)) * sqrt((α - 2.0) / α) : NaN
end

function kurtosis(d::Pareto)
    α = shape(d)
    α > 4.0 ? (6.0 * (α^3 + α^2 - 6.0 * α - 2.0)) / (α * (α - 3.0) * (α - 4.0)) : NaN
end

entropy(d::Pareto) = ((α, s) = params(d); log(s / α) + 1.0 / α + 1.0)


#### Evaluation

function pdf(d::Pareto, x::Float64)
    (α, s) = params(d)
    x >= s ? α * (s / x)^α * (1.0 / x) : 0.0
end

function logpdf(d::Pareto, x::Float64)
    (α, s) = params(d)
    x >= s ? log(α) + α * log(s) - (α + 1.0) * log(x) : -Inf 
end

function ccdf(d::Pareto, x::Float64)
    (α, s) = params(d)
    x >= s ? (s / x)^α : 1.0
end

cdf(d::Pareto, x::Float64) = 1.0 - ccdf(d, x)

function logccdf(d::Pareto, x::Float64)
    (α, s) = params(d)
    x >= s ? α * log(s / x) : 0.0
end

logcdf(d::Pareto, q::Float64) = log1p(-ccdf(d,q))

cquantile(d::Pareto, p::Float64) = d.s / p^(1.0 / d.α)
quantile(d::Pareto, p::Float64) = cquantile(d, 1.0 - p)


#### Sampling

rand(d::Pareto) = d.s * exp(randexp() / d.α)



