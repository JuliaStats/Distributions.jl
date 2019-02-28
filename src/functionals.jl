function getEndpoints(distr::UnivariateDistribution, epsilon::Real)
    (left,right) = map(x -> quantile(distr,x), (0,1))
    leftEnd = left!=-Inf ? left : quantile(distr, epsilon)
    rightEnd = right!=-Inf ? right : quantile(distr, 1-epsilon)
    (leftEnd, rightEnd)
end

function expectation(distr::ContinuousUnivariateDistribution, g::Function, epsilon::Real)
    f = x->pdf(distr,x)
    (leftEnd, rightEnd) = getEndpoints(distr, epsilon)
    integrate(x -> f(x)*g(x), leftEnd, rightEnd)
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

## Leave undefined until we've implemented a numerical integration procedure
# function entropy(distr::UnivariateDistribution)
#     pf = typeof(distr)<:ContinuousDistribution ? pdf : pmf
#     f = x -> pf(distr, x)
#     expectation(distr, x -> -log(f(x)))
# end

function kldivergence(P::UnivariateDistribution, Q::UnivariateDistribution)
    expectation(P, x -> let p = pdf(P,x); (p > 0)*log(p/pdf(Q,x)) end)
end
