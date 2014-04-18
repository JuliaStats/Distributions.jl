immutable Pareto <: ContinuousUnivariateDistribution
    scale::Float64
    shape::Float64
    function Pareto(sc::Real, sh::Real)
        sc > zero(sc) && sh > zero(sh) || error("shape and scale must be positive")
        new(float64(sc), float64(sh))
    end
end

Pareto() = Pareto(1.0, 1.0)
Pareto(scale::Real) = Pareto(scale, 1.0)

islowerbounded(::Union(Pareto, Type{Pareto})) = true
isupperbounded(::Union(Pareto, Type{Pareto})) = false
isbounded(::Union(Pareto, Type{Pareto})) = false

minimum(d::Pareto) = d.scale
maximum(d::Pareto) = Inf
insupport(d::Pareto, x::Number) = isfinite(x) && x >= d.scale

mean(d::Pareto) = d.shape > 1.0 ? (d.scale * d.shape) / (d.shape - 1.0) : Inf

median(d::Pareto) = d.scale * 2.0^(1.0/d.shape)

mode(d::Pareto) = d.scale

function var(d::Pareto)
    α = d.shape
    α > 2.0 ? (d.scale^2 * α) / ((α - 1.0)^2 * (α - 2.0)) : Inf
end

function skewness(d::Pareto)
    α = d.shape
    α > 3.0 ? ((2.0 * (1.0 + α)) / (α - 3.0)) * sqrt((α - 2.0) / α) : NaN
end


function kurtosis(d::Pareto)
    α = d.shape
    α > 4.0 ? (6.0 * (α^3 + α^2 - 6.0 * α - 2.0)) / (α * (α - 3.0) * (α - 4.0)) : NaN
end

entropy(d::Pareto) = log(d.scale / d.shape) + 1. / d.shape + 1.

function pdf(d::Pareto, q::Real)
    q >= d.scale ? d.shape * (d.scale/q)^d.shape / q : 0.0
end

ccdf(d::Pareto, q::Real) = q >= d.scale ? (d.scale / q)^d.shape : 1.0
logccdf(d::Pareto, q::Real) = q >= d.scale ? d.shape*log(d.scale / q) : 0.0
cdf(d::Pareto, q::Real) = 1.0 - ccdf(d,q)
logcdf(d::Pareto, q::Real) = log1p(-ccdf(d,q))

cquantile(d::Pareto, p::Real) = d.scale / p^(1.0 / d.shape)
quantile(d::Pareto, p::Real) = cquantile(d,1.0-p)

rand(d::Pareto) = d.scale*exp(rand(Exponential(1.0/d.shape)))

