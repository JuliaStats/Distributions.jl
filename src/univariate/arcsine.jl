##############################################################################
#
# REFERENCES: Using definition from Devroye, IX.7
#  This definition differs from definitions on Wikipedia and other sources,
#  where distribution is over [0, 1] rather than [-1, 1].
#
##############################################################################

immutable Arcsine <: ContinuousUnivariateDistribution
end

function cdf(d::Arcsine, x::Number)
    if x < -1.0
        return 0.0
    elseif x > 1.0
        return 1.0
    else
        return (2.0 / pi) * asin(sqrt((x + 1.0) / 2.0))
    end
end

entropy(d::Arcsine) = -log(2.0) / pi

function insupport(d::Arcsine, x::Number)
    if -1.0 <= x <= 1.0
        return true
    else
        return false
    end
end

mean(d::Arcsine) = 0.0

median(d::Arcsine) = 0.0

function pdf(d::Arcsine, x::Number)
    if insupport(d, x)
        return 1.0 / (pi * sqrt(1.0 - x^2))
    else
        return 0.0
    end
end

quantile(d::Arcsine, p::Real) = 2.0 * sin((pi / 2.0) * p)^2 - 1.0

rand(d::Arcsine) = sin(2.0 * pi * rand())

skewness(d::Arcsine) = 0.0

var(d::Arcsine) = 1.0 / 2.0
