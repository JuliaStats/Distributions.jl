immutable Pareto <: ContinuousUnivariateDistribution
    scale::Float64
    shape::Float64
    function Pareto(sc::Real, sh::Real)
        sc > zero(sc) && sh > zero(sh) || error("shape and scale must be positive")
        new(float64(sc), float64(sh))
    end
    Pareto() = new(1.0, 1.0)
end

Pareto(scale::Real) = Pareto(scale, 1.0)

cdf(d::Pareto, q::Real) = q >= d.scale ? 1.0 - (d.scale / q)^d.shape : 0.0

entropy(d::Pareto) = log(d.shape / d.scale) + 1. / d.scale + 1.

insupport(d::Pareto, x::Number) = isreal(x) && isfinite(x) && x > d.scale

function kurtosis(d::Pareto)
    a = d.shape
    a > 4.0 || error("Kurtosis undefined for Pareto w/ shape <= 4")
    (6.0 * (a^3 + a^2 - 6.0 * a - 2.0)) / (a * (a - 3.0) * (a - 4.0))
end

mean(d::Pareto) = d.shape <= 1.0 ? Inf : (d.scale * d.shape) / (d.scale - 1.0)

median(d::Pareto) = d.scale * 2.0^d.shape

mode(d::Pareto) = d.scale
modes(d::Pareto) = [d.scale]

function pdf(d::Pareto, q::Real)
    q >= d.scale ? (d.shape * d.scale^d.shape) / (q^(d.shape + 1.0)) : 0.0
end

quantile(d::Pareto, p::Real) = d.scale / (1.0 - p)^(1.0 / d.shape)

rand(d::Pareto) = d.shape / (rand()^(1.0 / d.scale))

function skewness(d::Pareto)
    a = d.shape
    a > 3.0 || error("Skewness undefined for Pareto w/ shape <= 3")
    ((2.0 * (1.0 + a)) / (a - 3.0)) * sqrt((a - 2.0) / a)
end

function var(d::Pareto)
    d.scale < 2.0 ? Inf : (d.shape^2 * d.scale) / ((d.scale - 1.0)^2 * (d.scale - 2.0))
end
