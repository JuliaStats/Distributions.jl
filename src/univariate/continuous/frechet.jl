# Fréchet Distribution

immutable Frechet <: ContinuousUnivariateDistribution
    α::Float64
    s::Float64

    function Frechet(α::Real, s::Real)
    	α > zero(α) && s > zero(s) || error("Both shape and scale must be positive")
    	new(float64(α), float64(s))
    end

    Frechet(α::Real) = Frechet(α, 1.0)
    Frechet() = new(1.0, 1.0)
end

@distr_support Frechet 0.0 Inf


#### Parameters

shape(d::Frechet) = d.α
scale(d::Frechet) = d.s
params(d::Frechet) = (d.α, d.s)


#### Statistics

mean(d::Frechet) = (α = d.α; α > 1.0 ? d.s * gamma(1.0 - 1.0 / α) : Inf)

median(d::Frechet) = d.s * logtwo^(-1.0 / d.α)

mode(d::Frechet) = (iα = -1.0/d.α; d.s * (1.0 - iα) ^ iα)

function var(d::Frechet)
    if d.α > 2.0
        iα = 1.0 / d.α    
        return d.s^2 * (gamma(1.0 - 2.0 * iα) - gamma(1.0 - iα)^2)
    else    
        return Inf
    end
end

function skewness(d::Frechet)
    if d.α > 3.0
        iα = 1.0 / d.α
        g1 = gamma(1.0 - iα)
        g2 = gamma(1.0 - 2.0 * iα)
        g3 = gamma(1.0 - 3.0 * iα)
        return (g3 - 3.0 * g2 * g1 + 2 * g1^3) / ((g2 - g1^2)^1.5)
    else
        return Inf
    end
end

function kurtosis(d::Frechet)
    if d.α > 3.0
        iα = 1.0 / d.α
        g1 = gamma(1.0 - iα)
        g2 = gamma(1.0 - 2.0 * iα)
        g3 = gamma(1.0 - 3.0 * iα)
        g4 = gamma(1.0 - 4.0 * iα)
        return (g4 - 4.0 * g3 * g1 + 3 * g2^2) / ((g2 - g1^2)^2) - 6.0
    else
        return Inf
    end    
end

function entropy(d::Frechet)
    const γ = 0.57721566490153286060  # γ is the Euler-Mascheroni constant
    1.0 + γ / d.α + γ + log(d.s / d.α)
end


#### Evaluation

function logpdf(d::Frechet, x::Float64)
    (α, s) = params(d)
    if x > 0.0
        z = s / x
        return log(α / s) + (1.0 + α) * log(z) - z^α
    else
        return -Inf
    end
end

pdf(d::Frechet, x::Float64) = exp(logpdf(d, x))

cdf(d::Frechet, x::Float64) = x > 0.0 ? exp(-((d.s / x) ^ d.α)) : 0.0
ccdf(d::Frechet, x::Float64) = x > 0.0 ? -expm1(-((d.s / x) ^ d.α)) : 1.0
logcdf(d::Frechet, x::Float64) = x > 0.0 ? -(d.s / x) ^ d.α : -Inf
logccdf(d::Frechet, x::Float64) = x > 0.0 ? log1mexp(-((d.s / x) ^ d.α)) : 0.0

quantile(d::Frechet, p::Float64) = d.s * (-log(p)) ^ (-1.0 / d.α)
cquantile(d::Frechet, p::Float64) = d.s * (-log1p(-p)) ^ (-1.0 / d.α)
invlogcdf(d::Frechet, lp::Float64) = d.s * (-lp)^(-1.0 / d.α)
invlogccdf(d::Frechet, lp::Float64) = d.s * (-log1mexp(lp))^(-1.0 / d.α)

function gradlogpdf(d::Frechet, x::Float64)
    (α, s) = params(d)
    insupport(Frechet, x) ? -(α + 1.0) / x + α * (s^α) * x^(-α-1.0)  : 0.0
end

## Sampling

rand(d::Frechet) = d.s * randexp() ^ (-1.0 / d.α)


