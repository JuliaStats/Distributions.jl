immutable Levy <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Levy(l::Real, s::Real)
        s >= zero(s) || error("scale must be non-negative")
        new(float64(l), float64(s))
    end
end

Levy(location::Real) = Levy(location, 1.0)
Levy() = Levy(0.0, 1.0)

@distr_support Levy d.location Inf

function entropy(d::Levy)
    c = d.scale
    (1.0 - 3.0 * digamma(1.0) + log(16.0 * pi * c * c)) / 2.0
end

mean(d::Levy) = Inf
var(d::Levy) = Inf
skewness(d::Levy) = NaN
kurtosis(d::Levy) = NaN

mode(d::Levy) = d.scale / 3.0 + d.location

function median(d::Levy)
    m, c = d.location, d.scale
    m + c / (2.0 * erfcinv(0.5)^2)
end

function pdf(d::Levy, x::Float64)
    m, c = d.location, d.scale
    (sqrt(c)/sqrt2π) * exp(-(c / (2.0 * (x - m)))) / (x - m)^1.5
end

cdf(d::Levy, x::Float64) = erfc(sqrt(d.scale / (2.0 * (x - d.location))))

quantile(d::Levy, p::Float64) = d.location + d.scale / (2.erfcinv(p)^2)

mgf(d::Levy, t::Real) = t == zero(t) ? 1.0 : NaN

function cf(d::Levy, t::Real)
    m, c = d.location, d.scale
    exp(im * m * t - sqrt(-2.0 * im * c * t))
end

rand(d::Levy) = d.location + 1 / rand(Normal(0.0, 1.0 / sqrt(d.scale)))^2
