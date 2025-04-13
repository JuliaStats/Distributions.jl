function expectation(g, distr::ContinuousUnivariateDistribution; kwargs...)
    return first(quadgk(x -> pdf(distr, x) * g(x), extrema(distr)...; kwargs...))
end

## Assuming that discrete distributions only take integer values.
function expectation(g, distr::DiscreteUnivariateDistribution; epsilon::Real=1e-10)
    mindist, maxdist = extrema(distr)
    # We want to avoid taking values up to infinity
    minval = isfinite(mindist) ? mindist : quantile(distr, epsilon)
    maxval = isfinite(maxdist) ? maxdist : quantile(distr, 1 - epsilon)
    return sum(x -> pdf(distr, x) * g(x), minval:maxval)
end

function expectation(g, distr::MultivariateDistribution; nsamples::Int=100, rng::AbstractRNG=default_rng())
    nsamples > 0 || throw(ArgumentError("number of samples should be > 0"))
    # We use a function barrier to work around type instability of `sampler(dist)`
    return mcexpectation(rng, g, sampler(distr), nsamples)
end

mcexpectation(rng, f, sampler, n) = sum(f, rand(rng, sampler) for _ in 1:n) / n

## Leave undefined until we've implemented a numerical integration procedure
# function entropy(distr::UnivariateDistribution)
#     pf = typeof(distr)<:ContinuousDistribution ? pdf : pmf
#     f = x -> pf(distr, x)
#     expectation(distr, x -> -log(f(x)))
# end

function kldivergence(p::Distribution{V}, q::Distribution{V}; kwargs...) where {V<:VariateForm}
    return expectation(p; kwargs...) do x
        logp = logpdf(p, x)
        return (logp > oftype(logp, -Inf)) * (logp - logpdf(q, x))
    end
end
