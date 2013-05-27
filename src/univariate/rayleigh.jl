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

insupport(d::Rayleigh, x::Number) = isreal(x) && isfinite(x) && 0.0 < x

kurtosis(d::Rayleigh) = d.scale^4 * (8.0 - ((3.0 * pi^2) / 4.0))

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

skewness(d::Rayleigh) = d.scale^3 * (pi - 3.0) * sqrt(pi / 2.0)

var(d::Rayleigh) = d.scale^2 * (2.0 - pi / 2.0)
