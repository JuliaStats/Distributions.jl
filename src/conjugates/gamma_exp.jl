# Conjuagtes for
#
#	Gamma - Exponential
#

function posterior(prior::Gamma, ss::ExponentialStats)
	return Gamma(prior.shape + ss.sw, 1.0 / (rate(prior) + ss.sx))
end

posterior_make(::Type{Exponential}, θ::Float64) = Exponential(1.0 / θ)

