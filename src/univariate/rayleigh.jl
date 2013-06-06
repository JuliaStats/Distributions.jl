##############################################################################
#
# Rayleigh distribution from Distributions Handbook
#
##############################################################################

immutable Rayleigh <: ContinuousUnivariateDistribution
    scale::Float64
    function Rayleigh(s::Real)
        if s > 0.0
            new(float64(s))
        else
            error("scale must be positive")
        end
    end
end

Rayleigh() = Rayleigh(1.0)

entropy(d::Rayleigh) = 1.0 + log(d.scale) - log(sqrt(2.0)) - digamma(1.0) / 2.0

insupport(d::Rayleigh, x::Number) = isreal(x) && isfinite(x) && 0.0 < x

kurtosis(d::Rayleigh) = -(6.0 * pi^2 - 24.0 * pi + 16.0) / (4.0 - pi)^2

mean(d::Rayleigh) = d.scale * sqrt(pi / 2.)

median(d::Rayleigh) = d.scale * sqrt(2. * log(2.))

function pdf(d::Rayleigh, x::Real)
    if insupport(d, x)
        return (x / (d.scale^2)) * exp(-(x^2)/(2.0 * (d.scale^2)))
    else
        return 0.0
    end
end

rand(d::Rayleigh) = d.scale * sqrt(-2.0 * log(rand()))

skewness(d::Rayleigh) = (2.0 * sqrt(pi) * (pi - 3.0)) / (4.0 - pi)^1.5

var(d::Rayleigh) = d.scale^2 * (2.0 - pi / 2.0)
