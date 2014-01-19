##############################################################################
#
# Symmetric triangular distribution from Distributions Handbook
#
##############################################################################

immutable TriangularDist <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function TriangularDist(l::Real, s::Real)
        s > zero(s) || error("scale must be positive")
        new(float64(l), float64(s))
    end
end

TriangularDist(location::Real) = TriangularDist(location, 1.0)
TriangularDist() = TriangularDist(0.0, 1.0)

function cdf(d::TriangularDist, x::Real)
    a, b, c = d.location - d.scale, d.location + d.scale, d.location
    if x <= a
        return 0.0
    elseif a <= x <= c
        return (x - a)^2 / ((b - a) * (c - a))
    elseif c < x <= b
        return 1.0 - (b - x)^2 / ((b - a) * (b - c))
    else
        return 1.0
    end
end

entropy(d::TriangularDist) = 0.5 + log(d.scale)

# support handling

isupperbounded(::Union(TriangularDist, Type{TriangularDist})) = true
islowerbounded(::Union(TriangularDist, Type{TriangularDist})) = true
isbounded(::Union(TriangularDist, Type{TriangularDist})) = true

minimum(d::TriangularDist) = d.location - d.scale
maximum(d::TriangularDist) = d.location + d.scale
insupport(d::TriangularDist, x::Real) = minimum(d) <= x <= maximum(d)


kurtosis(d::TriangularDist) = -0.6

mean(d::TriangularDist) = d.location

median(d::TriangularDist) = d.location

mode(d::TriangularDist) = d.location
modes(d::TriangularDist) = [d.location]

function pdf(d::TriangularDist, x::Real)
    if insupport(d, x)
        return -abs(x - d.location) / (d.scale^2) + 1.0 / d.scale
    else
        return 0.0
    end
end

function quantile(d::TriangularDist, p::Real)
    a, b, c = d.location - d.scale, d.location + d.scale, d.location
    if p <= 0.0
        return a
    elseif p < 0.5
        return a + sqrt(p * 2.0 * d.scale^2)
    elseif p >= 0.5
        return b -  sqrt((1.0 - p) * 2.0 * d.scale^2)
    else
        return b
    end
end

function rand(d::TriangularDist)
    ξ1, ξ2 = rand(), rand()
    return d.location + (ξ1 - ξ2) * d.scale
end

function skewness(d::TriangularDist)
    a = d.location - d.scale
    b = d.location + d.scale
    c = (b - a) / 2 + a
    den = sqrt(2.0) * (a + b - 2.0 * c) *
                      (2.0 * a - b - c) *
                      (a - 2.0 * b + c)
    num = 5.0 * (a^2 + b^2 + c^2 - a * b - a * c - b * c)^1.5
    return den / num
end

var(d::TriangularDist) = d.scale^2 / 6.0
