# Fallback functions for conjugates

posterior_canon(pri::Distribution, G::IncompleteFormulation, x) = posterior_canon(pri, suffstats(G, x))
posterior_canon(pri::Distribution, G::IncompleteFormulation, x, w) = posterior_canon(pri, suffstats(G, x, w))

posterior{P<:Distribution}(pri::P, ss::SufficientStats) = convert(P, posterior_canon(pri, ss)) 
posterior{P<:Distribution}(pri::P, G::IncompleteFormulation, x) = convert(P, posterior_canon(pri, G, x))
posterior{P<:Distribution}(pri::P, G::IncompleteFormulation, x, w) = convert(P, posterior_canon(pri, G, x, w))

posterior_rand(pri::Distribution, ss::SufficientStats) = rand(posterior_canon(pri, ss))
posterior_rand(pri::Distribution, G::IncompleteFormulation, x) = rand(posterior_canon(pri, G, x))
posterior_rand(pri::Distribution, G::IncompleteFormulation, x, w) = rand(posterior_canon(pri, G, x, w))

posterior_rand!(r::Array, pri::Distribution, ss::SufficientStats) = rand!(posterior_canon(pri, ss), r)
posterior_rand!(r::Array, pri::Distribution, G::IncompleteFormulation, x) = rand!(posterior_canon(pri, G, x), r)
posterior_rand!(r::Array, pri::Distribution, G::IncompleteFormulation, x, w) = rand!(posterior_canon(pri, G, x, w), r)

posterior_mode(pri::Distribution, ss::SufficientStats) = mode(posterior_canon(pri, ss))
posterior_mode(pri::Distribution, G::IncompleteFormulation, x) = mode(posterior_canon(pri, G, x))
posterior_mode(pri::Distribution, G::IncompleteFormulation, x, w) = mode(posterior_canon(pri, G, x, w))

fit_map(pri::Distribution, G::IncompleteFormulation, x) = complete(G, pri, posterior_mode(pri, G, x)...)
fit_map(pri::Distribution, G::IncompleteFormulation, x, w) = complete(G, pri, posterior_mode(pri, G, x, w)...)

posterior_randmodel(pri::Distribution, G::IncompleteFormulation, x) = complete(G, pri, posterior_rand(pri, G, x)...)
posterior_randmodel(pri::Distribution, G::IncompleteFormulation, x, w) = complete(G, pri, posterior_rand(pri, G, x, w)...)

