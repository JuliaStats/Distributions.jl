# Conjugates for 
#
#   Beta - Bernoulli
#   Beta - Binomial
#

posterior(prior::Beta, ss::BernoulliStats) = Beta(prior.alpha + ss.cnt1, prior.beta + ss.cnt0)

posterior(prior::Beta, ss::BinomialStats) = Beta(prior.alpha + ss.ns, prior.beta + (ss.ne * ss.n - ss.ns))

function posterior{T<:Real}(prior::Beta, ::Type{Binomial}, n::Integer, x::Array{T})
	posterior(prior, suffstats(Binomial, n, x))
end

function posterior{T<:Real}(prior::Beta, ::Type{Binomial}, n::Integer, x::Array{T}, w::Array{Float64})
	posterior(prior, suffstats(Binomial, n, x, w))
end


