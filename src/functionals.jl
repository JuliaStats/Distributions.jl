function getEndpoints(distr::UnivariateDistribution, epsilon::Real)
    (left,right) = map(x -> quantile(distr,x), (0,1))
    leftEnd = left!=-Inf ? left : quantile(distr, epsilon)
    rightEnd = right!=-Inf ? right : quantile(distr, 1-epsilon)
    (leftEnd, rightEnd)
end

function expectation(distr::ContinuousUnivariateDistribution, g::Function, epsilon::Real)
    f = x->pdf(distr,x)
    (leftEnd, rightEnd) = getEndpoints(distr, epsilon)
    quadgk(x -> f(x)*g(x), leftEnd, rightEnd)[1]
end

## Assuming that discrete distributions only take integer values.
function expectation(distr::DiscreteUnivariateDistribution, g::Function, epsilon::Real)
    f = x->pdf(distr,x)
    (leftEnd, rightEnd) = getEndpoints(distr, epsilon)
    sum(x -> f(x)*g(x), leftEnd:rightEnd)
end

function expectation(distr::UnivariateDistribution, g::Function)
    expectation(distr, g, 1e-10)
end

function expectation(distr::MultivariateDistribution, g::Function; nsamples::Int=100, rng::AbstractRNG=GLOBAL_RNG)
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

function kldivergence(P::Distribution{V}, Q::Distribution{V}; kwargs...) where {V<:VariateForm}
    expectation(P, x -> let p = pdf(P, x); (p > 0) * log(p / pdf(Q, x)) end)
end

function kldivergence(P::MultivariateDistribution, Q::MultivariateDistribution; kwargs...)
    expectation(P, x -> let p = pdf(P, x); (p > 0) * log(p / pdf(Q, x)) end; kwargs...)
end
