##############################################################################
#
# Rayleigh distribution from Distributions Handbook
#
##############################################################################

immutable Rayleigh <: ContinuousUnivariateDistribution
    scale::Float64
    function Rayleigh(s::Real)
        s > zero(s) || error("scale must be positive")
        new(float64(s))
    end
    Rayleigh() = new(1.0)
end

cdf(d::Rayleigh, x::Real) = 1.0 - exp(-x^2 / (2.0 * d.scale^2))

entropy(d::Rayleigh) = 1.0 + log(d.scale) - log(sqrt(2.0)) - digamma(1.0) / 2.0

insupport(::Rayleigh, x::Real) = zero(x) < x < Inf
insupport(::Type{Rayleigh}, x::Real) = zero(x) < x < Inf

kurtosis(d::Rayleigh) = -(6.0 * pi^2 - 24.0 * pi + 16.0) / (4.0 - pi)^2

mean(d::Rayleigh) = d.scale * sqrt(pi / 2.)

median(d::Rayleigh) = d.scale * sqrt(2. * log(2.))

mode(d::Rayleigh) = d.scale
modes(d::Rayleigh) = [d.scale]

function pdf(d::Rayleigh, x::Real)
    insupport(d, x) ? (x / (d.scale^2)) * exp(-(x^2)/(2.0 * (d.scale^2))) : 0.0
end

quantile(d::Rayleigh, p::Real) = sqrt(-2.0 * d.scale^2 * log(1.0 - p))

rand(d::Rayleigh) = d.scale * sqrt(-2.0 * log(rand()))

skewness(d::Rayleigh) = (2.0 * sqrt(pi) * (pi - 3.0)) / (4.0 - pi)^1.5

var(d::Rayleigh) = d.scale^2 * (2.0 - pi / 2.0)
