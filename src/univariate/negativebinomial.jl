# NegativeBinomial is the distribution of the number of failures
# before the size'th success in a sequence of Bernoulli trials.
# We do not enforce integer size, as the distribution is well defined
# for non-integers, and this can be useful for e.g. overdispersed
# discrete survival times.

immutable NegativeBinomial <: DiscreteUnivariateDistribution
    size::Float64
    prob::Float64
    function NegativeBinomial(s::Real, p::Real)
    	if 0.0 < p <= 1.0
    		if s >= 0.0
    			new(float64(s), float64(p))
    		else
    			error("size must be non-negative")
			end
		else
			error("prob must be in (0, 1]")
		end
    end
end

@_jl_dist_2p NegativeBinomial nbinom

insupport(d::NegativeBinomial, x::Number) = isinteger(x) && 0.0 <= x

function mgf(d::NegativeBinomial, t::Real)
    r, p = d.size, d.prob
    return ((1.0 - p) * exp(t))^r / (1.0 - p * exp(t))^r
end

function cf(d::NegativeBinomial, t::Real)
    r, p = d.size, d.prob
    return ((1.0 - p) * exp(im * t))^r / (1.0 - p * exp(im * t))^r
end
