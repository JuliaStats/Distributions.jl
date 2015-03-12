immutable Pareto <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64

    function Pareto(α::Real, β::Real)
        (α > zero(α) && β > zero(β)) || error("Pareto: shape and scale must be positive")
        @compat new(Float64(α), Float64(β))
    end

    Pareto(α::Real) = Pareto(α, 1.0)
    Pareto() = new(1.0, 1.0)
end

@distr_support Pareto d.β Inf


#### Parameters

shape(d::Pareto) = d.α
scale(d::Pareto) = d.β

params(d::Pareto) = (d.α, d.β)


#### Statistics

mean(d::Pareto) = ((α, β) = params(d); α > 1.0 ? α * β / (α - 1.0) : Inf)
median(d::Pareto) = ((α, β) = params(d); β * 2.0 ^ (1.0 / α))
mode(d::Pareto) = d.β

function var(d::Pareto)
    (α, β) = params(d)
    α > 2.0 ? (β^2 * α) / ((α - 1.0)^2 * (α - 2.0)) : Inf
end

function skewness(d::Pareto)
    α = shape(d)
    α > 3.0 ? ((2.0 * (1.0 + α)) / (α - 3.0)) * sqrt((α - 2.0) / α) : NaN
end

function kurtosis(d::Pareto)
    α = shape(d)
    α > 4.0 ? (6.0 * (α^3 + α^2 - 6.0 * α - 2.0)) / (α * (α - 3.0) * (α - 4.0)) : NaN
end

entropy(d::Pareto) = ((α, β) = params(d); log(β / α) + 1.0 / α + 1.0)


#### Evaluation

function pdf(d::Pareto, x::Float64)
    (α, β) = params(d)
    x >= β ? α * (β / x)^α * (1.0 / x) : 0.0
end

function logpdf(d::Pareto, x::Float64)
    (α, β) = params(d)
    x >= β ? log(α) + α * log(β) - (α + 1.0) * log(x) : -Inf 
end

function ccdf(d::Pareto, x::Float64)
    (α, β) = params(d)
    x >= β ? (β / x)^α : 1.0
end

cdf(d::Pareto, x::Float64) = 1.0 - ccdf(d, x)

function logccdf(d::Pareto, x::Float64)
    (α, β) = params(d)
    x >= β ? α * log(β / x) : 0.0
end

logcdf(d::Pareto, x::Float64) = log1p(-ccdf(d, x))

cquantile(d::Pareto, p::Float64) = d.β / p^(1.0 / d.α)
quantile(d::Pareto, p::Float64) = cquantile(d, 1.0 - p)


#### Sampling

rand(d::Pareto) = d.β * exp(randexp() / d.α)



