immutable Geometric <: DiscreteUnivariateDistribution
    prob::Float64
    function Geometric(p::Real)
    	if 0.0 < p < 1.0
    		new(float64(p))
    	else
    		error("prob must be in (0, 1)")
    	end
    end
end

Geometric() = Geometric(0.5) # Flips of a fair coin

@_jl_dist_1p Geometric geom

function cdf(d::Geometric, q::Real)
    q < 0.0 ? 0.0 : -expm1(log1p(-d.prob) * (floor(q) + 1.0))
end

function ccdf(d::Geometric, q::Real)
    q < 0.0 ? 1.0 : exp(log1p(-d.prob) * (floor(q + 1e-7) + 1.0))
end

entropy(d::Geometric) = (-xlogx(1.0 - d.prob) - xlogx(d.prob)) / d.prob

insupport(::Geometric, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{Geometric}, x::Real) = isinteger(x) && zero(x) <= x

kurtosis(d::Geometric) = 6.0 + d.prob^2 / (1.0 - d.prob)

mean(d::Geometric) = (1.0 - d.prob) / d.prob

function median(d::Geometric)
    iceil(-1.0 / log(2.0, 1.0 - d.prob)) - 1
end

modes(d::Geometric) = [0]

function mgf(d::Geometric, t::Real)
    p = d.prob
    if t >= -log(1.0 - p)
        error("MGF does not exist for all t")
    end
    return (p * exp(t)) / (1.0 - (1.0 - p) * exp(t))
end

function cf(d::Geometric, t::Real)
    p = d.prob
    return (p * exp(im * t)) / (1.0 - (1.0 - p) * exp(im * t))
end

skewness(d::Geometric) = (2.0 - d.prob) / sqrt(1.0 - d.prob)

var(d::Geometric) = (1.0 - d.prob) / d.prob^2

fit_mle(::Type{Geometric}, x::Array) = Geometric(1.0 / mean(x))
