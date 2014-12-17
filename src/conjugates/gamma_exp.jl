# Conjuagtes for
#
#	Gamma - Exponential
#

function posterior_canon(prior::Gamma, ss::ExponentialStats)
	return Gamma(shape(prior) + ss.sw, 1.0 / (rate(prior) + ss.sx))
end

complete(G::Type{Exponential}, pri::Gamma, θ::Float64) = Exponential(1.0 / θ)
