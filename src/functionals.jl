function support(distr::DiscreteUnivariateDistribution, epsilon::Real)
    (left,right) = map(x -> quantile(distr,x), (0,1))
    leftEnd = left!=-Inf ? left : quantile(distr, epsilon)
    rightEnd = right!=-Inf ? right : quantile(distr, 1-epsilon)
    leftEnd:rightEnd
end
function support(distr::ContinuousUnivariateDistribution, epsilon::Real)
    (left,right) = map(x -> quantile(distr,x), (0,1))
    leftEnd = left!=-Inf ? left : quantile(distr, epsilon)
    rightEnd = right!=-Inf ? right : quantile(distr, 1-epsilon)
    RealInterval(leftEnd, rightEnd)
end


function expectation(distr::ContinuousUnivariateDistribution, g::Function, epsilon::Real=1e-10)
    s = support(distr, epsilon)
    integrate(x -> pdf(distr, x)*g(x), s.lb, s.ub)
end

## Not assuming that discrete distributions only take integer values.
function expectation(distr::DiscreteUnivariateDistribution, g::Function, epsilon::Real = 0)
    sum(x -> pdf(distr, x)*g(x), support(distr, epsilon))
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
