# Conjugates for 
#
#   Beta - Bernoulli
#   Beta - Binomial
#

posterior_canon(pri::Beta, ss::BernoulliStats) = Beta(pri.alpha + ss.cnt1, pri.beta + ss.cnt0)
posterior_canon(pri::Beta, ss::BinomialStats) = Beta(pri.alpha + ss.ns, pri.beta + (ss.ne * ss.n - ss.ns))

complete(G::Type{Bernoulli}, pri::Beta, p::Float64) = Bernoulli(p)

# specialized fit_map and posterior_randmodel methods for Binomial 
#
# n is needed to create a Binomial distribution (which can not be provided through complete)
#

fit_map(pri::Beta, ss::BinomialStats) = Binomial(ss.n, posterior_mode(pri, ss))
fit_map(pri::Beta, G::Type{Binomial}, data::(Int, Array)) = fit_map(pri, suffstats(G, data))
fit_map(pri::Beta, G::Type{Binomial}, data::(Int, Array), w::Array) = fit_map(pri, suffstats(G, data, w))

posterior_randmodel(pri::Beta, ss::BinomialStats) = Binomial(ss.n, posterior_rand(pri, ss))

function posterior_randmodel(pri::Beta, G::Type{Binomial}, data::(Int, Array))
	posterior_randmodel(pri, suffstats(G, data))
end

function posterior_randmodel(pri::Beta, G::Type{Binomial}, data::(Int, Array), w::Array)
	posterior_randmodel(pri, suffstats(G, data, w))
end


