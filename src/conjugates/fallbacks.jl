# Fallback functions for conjugates

posterior(pri::Distribution, G::IncompleteFormulation, x) = posterior(pri, suffstats(G, x))
posterior(pri::Distribution, G::IncompleteFormulation, x, w) = posterior(pri, suffstats(G, x, w))

posterior_rand(pri::Distribution, s::SufficientStats) = rand(posterior(pri, s))
posterior_rand(pri::Distribution, G::IncompleteFormulation, x) = rand(posterior(pri, G, x))
posterior_rand(pri::Distribution, G::IncompleteFormulation, x, w) = rand(posterior(pri, G, x, w))

posterior_rand!(r::Array, pri::Distribution, s::SufficientStats) = rand!(posterior(pri, s), r)
posterior_rand!(r::Array, pri::Distribution, G::IncompleteFormulation, x) = rand!(posterior(pri, G, x), r)
posterior_rand!(r::Array, pri::Distribution, G::IncompleteFormulation, x, w) = rand!(posterior(pri, G, x, w), r)

posterior_mode(pri::Distribution, s::SufficientStats) = mode(posterior(pri, s))
posterior_mode(pri::Distribution, G::IncompleteFormulation, x) = mode(posterior(pri, G, x))
posterior_mode(pri::Distribution, G::IncompleteFormulation, x, w) = mode(posterior(pri, G, x, w))

posterior_make{D<:Distribution}(::Type{D}, θ) = D(θ) 
fit_map{D<:Distribution}(pri::Distribution, ::Type{D}, x) = posterior_make(D, posterior_mode(pri, D, x))
fit_map{D<:Distribution}(pri::Distribution, ::Type{D}, x, w) = posterior_make(D, posterior_mode(pri, D, x, w))

posterior_sample{D<:Distribution}(pri::Distribution, G::Type{D}, x) = D(rand(posterior(pri, G, x))...)
posterior_sample{D<:Distribution}(pri::Distribution, G::Type{D}, x, w) = D(rand(posterior(pri, G, x, w))...)
posterior_sample{D<:Distribution,G<:Distribution}(post::D, ::Type{G}) = G(rand(post)...)

