# NegativeBinomial is the distribution of the number of failures
# before the r-th success in a sequence of Bernoulli trials.
# We do not enforce integer size, as the distribution is well defined
# for non-integers, and this can be useful for e.g. overdispersed
# discrete survival times.

immutable NegativeBinomial <: DiscreteUnivariateDistribution
    r::Float64
    prob::Float64

    function NegativeBinomial(r::Real, p::Real)
        zero(p) < p <= one(p) || error("prob must be in (0, 1].")
        zero(r) < r || error("r must be positive.")
        new(float64(r), float64(p))
    end

    NegativeBinomial() = new(1.0, 0.5)
end

insupport(::NegativeBinomial, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{NegativeBinomial}, x::Real) = isinteger(x) && zero(x) <= x

function mean(d::NegativeBinomial)
    p = d.prob
    (1.0 - p) * d.r / p
end

function mode(d::NegativeBinomial)
    p = d.prob
    ifloor((1.0 - p) * (d.r - 1.) / p)
end
modes(d::NegativeBinomial) = [mode(d)]

function var(d::NegativeBinomial)
    p = d.prob
    (1.0 - p) * d.r / (p * p)
end

function std(d::NegativeBinomial)
    p = d.prob
    sqrt((1.0 - p) * d.r) / p
end

function skewness(d::NegativeBinomial)
    p = d.prob
    (2.0 - p) / sqrt((1.0 - p) * d.r)
end

function kurtosis(d::NegativeBinomial)
    p = d.prob
    6.0 / d.r + (p * p) / ((1.0 - p) * d.r)
end


function pdf(d::NegativeBinomial, x::Real)
    if !insupport(d,x)
        return 0.0
    end
    r, p = d.r, d.prob
    if x == 0
        return exp(r*log1p(-p))
    end
    q = 1.0-p
    n = x+r
    sqrt(r/(2.0*pi*x*n)) * exp((lstirling(n) - lstirling(x) - lstirling(r))
                             + x*logmxp1(n*p/x) + r*logmxp1(n*q/r))
end
function logpdf(d::NegativeBinomial, x::Real)
    if !insupport(d,x)
        return -Inf
    end
    r, p = d.r, d.prob
    if x == 0
        return r*log1p(-p)
    end
    q = 1.0-p
    n = x+r
    (lstirling(n) - lstirling(x) - lstirling(r)) +
    x*logmxp1(n*p/x) + r*logmxp1(n*q/r) + 0.5*(log(r/(x*n))-log2π)
end

function cdf(d::NegativeBinomial, x::Real)
    if x <= 0 return 0.0 end
    return bratio(d.r, floor(x)+1.0, d.prob)
end

function ccdf(d::NegativeBinomial, x::Real)
    if x <= 0 return 1.0 end
    return bratio(floor(x)+1.0, d.r, 1.0-d.prob)
end



function mgf(d::NegativeBinomial, t::Real)
    r, p = d.r, d.prob
    return ((1.0 - p) * exp(t))^r / (1.0 - p * exp(t))^r
end

function cf(d::NegativeBinomial, t::Real)
    r, p = d.r, d.prob
    return ((1.0 - p) * exp(im * t))^r / (1.0 - p * exp(im * t))^r
end

function rand(d::NegativeBinomial)
    lambda = rand(Gamma(d.r, (1-d.prob)/d.prob))
    rand(Poisson(lambda))
end