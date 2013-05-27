##############################################################################
#
# Symmetric triangular distribution from Distributions Handbook
#
##############################################################################

immutable Triangular <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Triangular(l::Real, s::Real)
        if s > 0.0
            new(float64(l), float64(s))
        else
            error("scale must be positive")
        end
    end
end

Triangular(location::Real) = Triangular(location, 1.0)
Triangular() = Triangular(0.0, 1.0)

function insupport(d::Triangular, x::Number)
    return isreal(x) && isfinite(x) &&
           d.location - d.scale <= x <= d.location + d.scale
end

kurtosis(d::Triangular) = d.scale^4 / 15.0

mean(d::Triangular) = d.location

median(d::Triangular) = d.location

modes(d::Triangular) = [d.location]

function pdf(d::Triangular, x::Real)
    if insupport(d, x)
        return -abs(x - d.location) / (d.scale^2) + 1.0 / d.scale
    else
        return 0.0
    end
end

function rand(d::Triangular)
    xi1, xi2 = rand(), rand()
    return d.location + (xi1 - xi2) * d.scale
end

skewness(d::Triangular) = 0.0

var(d::Triangular) = d.scale^2 / 6.0
