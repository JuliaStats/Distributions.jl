immutable Levy <: ContinuousUnivariateDistribution
    location::Float64
    scale::Float64
    function Levy(l::Real, s::Real)
        if s < 0.0
        	error("scale must be non-negative")
        else
        	new(float64(l), float64(s))
        end
    end
end

Levy(location::Real) = Levy(location, 1.0)
Levy() = Levy(0.0, 1.0)

cdf(d::Levy, x::Real) = erfc(sqrt(d.scale / (2.0 * (x - d.location))))

function entropy(d::Levy)
	c = d.scale
	return (1.0 - 3.0 * digamma(1.0) + log(16.0 * pi * c * c)) / 2.0
end

insupport(d::Levy, x::Real) = d.location <= x && isfinite(x)

mean(d::Levy) = Inf
var(d::Levy) = Inf
skewness(d::Levy) = NaN
kurtosis(d::Levy) = NaN


function median(d::Levy)
	m, c = d.location, d.scale
	return m + c / (2.0 * erfcinv(0.5)^2)
end

function mgf(d::Levy, t::Real)
	error("MGF is undefined for Levy distributions")
end

function cf(d::Levy, t::Real)
	m, c = d.location, d.scale
	return exp(im * m * t - sqrt(-2.0 * im * c * t))
end

modes(d::Levy) = [d.scale / 3.0 + d.location]

function pdf(d::Levy, x::Real)
	m, c = d.location, d.scale
	return sqrt(c / (2.0 * pi)) *
	       exp(-(c / (2.0 * (x - m)))) / (x - m)^1.5
end

quantile(d::Levy, p::Real) = d.location + d.scale / (2.0 * erfcinv(p)^2)

function rand(d::Levy)
	m, c = d.location, d.scale
	return m + 1 / rand(Normal(0.0, 1.0 / sqrt(c)))^2
end
