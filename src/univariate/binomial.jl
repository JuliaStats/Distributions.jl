immutable Binomial <: DiscreteUnivariateDistribution
    size::Int
    prob::Float64
    function Binomial(n::Real, p::Real)
    	if n <= 0
	    	error("size must be positive")
	    else
	    	if 0.0 <= p <= 1.0
	    		new(int(n), float64(p))
	    	else
	    		error("prob must be in [0, 1]")
			end
	    end
	end
end

Binomial(size::Integer) = Binomial(size, 0.5)
Binomial() = Binomial(1, 0.5)

@_jl_dist_2p Binomial binom

entropy(d::Binomial) = d.size * (-xlogx(1.0 - d.prob) - xlogx(d.prob))

insupport(d::Binomial, x::Number) = isinteger(x) && 0 <= x <= d.size

kurtosis(d::Binomial) = (1.0 - 6.0 * d.prob * (1.0 - d.prob)) / var(d)

mean(d::Binomial) = d.size * d.prob

median(d::Binomial) = iround(d.size * d.prob)

# TODO: May need to subtract 1 sometimes
modes(d::Binomial) = iround((d.size + 1.0) * d.prob)

function mgf(d::Binomial, t::Real)
	n, p = d.size, d.prob
	return (1.0 - p + p * exp(t))^n
end

function cf(d::Binomial, t::Real)
	n, p = d.size, d.prob
	return (1.0 - p + p * exp(im * t))^n
end

modes(d::Binomial) = iround([d.size * d.prob])

# TODO: rand() is totally screwed up

skewness(d::Binomial) = (1.0 - 2.0 * d.prob) / std(d)

var(d::Binomial) = d.size * d.prob * (1.0 - d.prob)

function fit(::Type{Binomial}, x::Real, n::Real)
    if x > n || x < 0
        error("For binomial observations, x must lie in [0, n]")
    end
    return Binomial(int(n), x / n)
end
