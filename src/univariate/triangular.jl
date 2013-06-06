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

entropy(d::Triangular) = 0.5 + log(d.scale)

function insupport(d::Triangular, x::Number)
    return isreal(x) && isfinite(x) &&
           d.location - d.scale <= x <= d.location + d.scale
end

kurtosis(d::Triangular) = -0.6

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
    ξ1, ξ2 = rand(), rand()
    return d.location + (ξ1 - ξ2) * d.scale
end

function skewness(d::Triangular)
    a = d.location - d.scale
    b = d.location + d.scale
    c = (b - a) / 2 + a
    den = sqrt(2.0) * (a + b - 2.0 * c) *
                      (2.0 * a - b - c) *
                      (a - 2.0 * b + c)
    num = 5.0 * (a^2 + b^2 + c^2 - a * b - a * c - b * c)^1.5
    return den / num
end

var(d::Triangular) = d.scale^2 / 6.0
