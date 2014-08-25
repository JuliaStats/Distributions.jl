# Triangular distribution
immutable TriangularDist <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
    c::Float64
    function TriangularDist(a::Real, b::Real, c::Real)
        a < b || error("a<b must be true")
        a <= c <= b || error("a<=c<=b must be true")
        new(float64(a), float64(b), float64(c))
    end
end

## Support
isupperbounded(::Union(TriangularDist, Type{TriangularDist})) = true
islowerbounded(::Union(TriangularDist, Type{TriangularDist})) = true
isbounded(::Union(TriangularDist, Type{TriangularDist})) = true

minimum(d::TriangularDist) = d.a
maximum(d::TriangularDist) = d.b
insupport(d::TriangularDist, x::Real) = minimum(d) <= x <= maximum(d)

## Properties
mean(d::TriangularDist) = (d.a + d.b + d.c) / 3.0
median(d::TriangularDist) = d.c >= (d.a+d.b)/2.0 ?
    d.a + sqrt((d.b-d.a)*(d.c-d.a))/sqrt(2.0) :
    d.b - sqrt((d.b-d.a)*(d.b-d.c))/sqrt(2.0)
mode(d::TriangularDist) = d.c

var(d::TriangularDist) = (d.a^2.0 + d.b^2.0 + d.c^2.0-d.a*d.b-d.a*d.c-d.b*d.c)/18.0
skewness(d::TriangularDist) = sqrt(2.0)*(d.a+d.b-2.0d.c)*(2d.a-d.b-d.c)*(d.a-2d.b+d.c)/5.0/(d.a^2.0 + d.b^2.0 + d.c^2.0-d.a*d.b-d.a*d.c-d.b*d.c)^(3.0/2.0)
kurtosis(d::TriangularDist) = -0.6

entropy(d::TriangularDist) = 0.5 + log((d.b-d.a)/2.0)

## Functions
function pdf(d::TriangularDist, x::Real)
    if d.a<=x<=d.c
        return 2.0*(x-d.a)/(d.b-d.a)/(d.c-d.a)
    elseif d.c<x<=d.b
        return 2.0*(d.b-x)/(d.b-d.a)/(d.b-d.c)
    else
        return zero(x)
    end
end

function cdf(d::TriangularDist, x::Real)
    if x<d.a
        return zero(x)
    elseif d.a<=x<=d.c
        return (x-d.a)^2/(d.b-d.a)/(d.c-d.a)
    elseif d.c<x<=d.b
        return 1.0-(d.b-x)^2/(d.b-d.a)/(d.b-d.c)
    else
        return one(x)
    end
end

function quantile(d::TriangularDist, p::Real)
    @checkquantile p begin
        if p <= (d.c-d.a)/(d.b-d.a)
            return d.a + sqrt((d.b-d.a)*(d.c-d.a)*p)
        else
            return d.b - sqrt((d.b-d.a)*(d.b-d.c)*(1-p))
        end
    end
end

function mgf(d::TriangularDist, t::Real)
    if t==zero(t)
        return one(t)
    else
        nominator = (d.b-d.c)*exp(d.a*t)-(d.b-d.a)*exp(d.c*t)+(d.c-d.a)*exp(d.b*t)
        denominator = (d.b-d.a)*(d.c-d.a)*(d.b-d.c)*t^2
        return 2*nominator/denominator
    end
end

function cf(d::TriangularDist, t::Real)
    # Is this correct?
    if t==zero(t)
        return one(t)
    else
        nominator = (d.b-d.c)*exp(im*d.a*t)-(d.b-d.a)*exp(im*d.c*t)+(d.c-d.a)*exp(im*d.b*t)
        denominator = (d.b-d.a)*(d.c-d.a)*(d.b-d.c)*t^2
        return -2*nominator/denominator
    end
end

## Sampling
function rand(d::TriangularDist)
    u = rand()

    if u < (d.c-d.a)/(d.b-d.a)
        return d.a + sqrt(u*(d.b-d.a)*(d.c-d.a))
    else
        return d.b - sqrt((1.0-u)*(d.b-d.a)*(d.b-d.c))
    end
end
