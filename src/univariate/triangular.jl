# Symmetric triangular distribution
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

## Support
isupperbounded(::Union(TriangularDist, Type{TriangularDist})) = true
islowerbounded(::Union(TriangularDist, Type{TriangularDist})) = true
isbounded(::Union(TriangularDist, Type{TriangularDist})) = true

minimum(d::TriangularDist) = d.location - d.scale
maximum(d::TriangularDist) = d.location + d.scale
insupport(d::TriangularDist, x::Real) = minimum(d) <= x <= maximum(d)

## Properties
mean(d::TriangularDist) = d.location
median(d::TriangularDist) = d.location
mode(d::TriangularDist) = d.location

var(d::TriangularDist) = d.scale^2 / 6.0
skewness(d::TriangularDist) = 0.0
kurtosis(d::TriangularDist) = -0.6

entropy(d::TriangularDist) = 0.5 + log(d.scale)

## Functions
function pdf(d::TriangularDist, x::Real)
    if insupport(d, x)
        (1.0 - abs(x-d.location)/d.scale) / d.scale
    else
        0.0
    end
end

function logpdf(d::TriangularDist, x::Real)
    if insupport(d, x)
        log1p(-abs(x-d.location)/d.scale) -log(d.scale)
    else
        -Inf
    end
end

function cdf(d::TriangularDist, x::Real)
    u = (x - d.location) / d.scale
    if u <= -1.0
        0.0
    elseif u <= 0.0
        0.5*(1.0 + u)^2
    elseif u <= 1.0
        1.0 - 0.5*(1.0 - u)^2
    else
        1.0
    end
end
function ccdf(d::TriangularDist, x::Real)
    u = (x - d.location) / d.scale
    if u <= -1.0
        1.0
    elseif u <= 0.0
        1.0 - 0.5*(1.0 + u)^2
    elseif u <= 1.0
        0.5*(1.0 - u)^2
    else
        0.0
    end
end

function logcdf(d::TriangularDist, x::Real)
    u = (x - d.location) / d.scale
    if u <= -1.0
        -Inf
    elseif u <= 0.0
        loghalf + 2.0*log1p(u)
    elseif u <= 1.0
        log1p(-0.5*(1.0 - u)^2)
    else
        0.0
    end
end
function logccdf(d::TriangularDist, x::Real)
    u = (x - d.location) / d.scale
    if u <= -1.0
        0.0
    elseif u <= 0.0
        log1p(- 0.5*(1.0 + u)^2)
    elseif u <= 1.0
        loghalf + 2.0*log1p(-u)
    else
        -Inf
    end
end

function quantile(d::TriangularDist, p::Real)
    @checkquantile p begin
        if p < 0.5
            d.location - d.scale*(1.0 - sqrt(2.0*p))
        else
            d.location + d.scale*(1.0 - sqrt(2.0*(1.0-p)))
        end
    end
end
function cquantile(d::TriangularDist, p::Real)
    @checkquantile p begin
        if p > 0.5
            d.location - d.scale*(1.0 - sqrt(2.0*(1.0-p)))
        else
            d.location + d.scale*(1.0 - sqrt(2.0*p))
        end
    end
end
function invlogcdf(d::TriangularDist, lp::Real)
    @checkinvlogcdf lp begin
        if lp < loghalf
            d.location + d.scale*expm1(0.5*(lp-loghalf))
        else
            d.location + d.scale*(1.0 - sqrt(-2.0*expm1(lp)))
        end
    end
end
function invlogccdf(d::TriangularDist, lp::Real)
    @checkinvlogcdf lp begin
        if lp > loghalf
            d.location - d.scale*(1.0 - sqrt(-2.0*expm1(lp)))
        else
            d.location - d.scale*expm1(0.5*(lp-loghalf))
        end
    end
end

function mgf(d::TriangularDist, t::Real)
    a = d.scale*t
    4.0*exp(d.location*t)*(sinh(0.5*a)/a)^2
end
function cf(d::TriangularDist, t::Real)
    a = d.scale*t
    4.0*exp(im*d.location*t)*(sin(0.5*a)/a)^2
end

## Sampling
function rand(d::TriangularDist)
    ξ1, ξ2 = rand(), rand()
    return d.location + (ξ1 - ξ2) * d.scale
end
