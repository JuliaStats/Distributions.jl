immutable DiscreteUniform <: DiscreteUnivariateDistribution
    a::Int
    b::Int
    function DiscreteUniform(a::Real, b::Real)
        ia = int(a); ib = int(b)
        ia < ib || error("int(a) must be less than int(b)")
        new(ia, ib)
    end
end

DiscreteUniform(b::Integer) = DiscreteUniform(0, b)
DiscreteUniform() = DiscreteUniform(0, 1)

function cdf(d::DiscreteUniform, k::Real)
    k < d.a ? 0. : (k > d.b ? 1. : (ifloor(k) - d.a + 1.0) / (d.b - d.a + 1.0))
end

entropy(d::DiscreteUniform) = log(d.b - d.a + 1.0)

function kurtosis(d::DiscreteUniform)
    n = d.b - d.a + 1.0
    return -(6.0 / 5.0) * (n^2 + 1.0) / (n^2 - 1.0)
end

mean(d::DiscreteUniform) = (d.a + d.b) / 2.0

median(d::DiscreteUniform) = (d.a + d.b) / 2.0

function mgf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    (exp(t * a) - exp(t * (b + 1))) / ((b - a + 1.0) * (1.0 - exp(t)))
end

function cf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    (exp(im * t * a) - exp(im * t * (b + 1))) / ((b - a + 1.0) * (1.0 - exp(im * t)))
end

mode(d::DiscreteUniform) = d.a
modes(d::DiscreteUniform) = [d.a:d.b]

function pdf(d::DiscreteUniform, x::Real)
    insupport(d, x) ? (1.0 / (d.b - d.a + 1)) : 0.0
end

function quantile(d::DiscreteUniform, p::Real)
    d.a + ifloor(p * (d.b - d.a + 1))
end

rand(d::DiscreteUniform) = randi(d.a, d.b)

skewness(d::DiscreteUniform) = 0.0

var(d::DiscreteUniform) = ((d.b - d.a + 1.0)^2 - 1.0) / 12.0

### handling support

isupperbounded(::Union(DiscreteUniform, Type{DiscreteUniform})) = true
islowerbounded(::Union(DiscreteUniform, Type{DiscreteUniform})) = true
isbounded(::Union(DiscreteUniform, Type{DiscreteUniform})) = true

minimum(d::DiscreteUniform) = d.a
maximum(d::DiscreteUniform) = d.b
support(d::DiscreteUniform) = d.a:d.b

insupport(d::DiscreteUniform, x::Number) = isinteger(x) && d.a <= x <= d.b

# Fit model

function fit_mle{T <: Real}(::Type{DiscreteUniform}, x::Array{T})
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end

    xmin = xmax = x[1]
    for i = 2:length(x)
        xi = x[i]
        if xi < xmin
            xmin = xi
        elseif xi > xmax
            xmax = xi
        end
    end

    DiscreteUniform(xmin, xmax)
end
