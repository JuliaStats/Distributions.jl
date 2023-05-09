function _check_tail(tail::Symbol)
    if tail !== :both && tail !== :left && tail !== :right
        throw(ArgumentError("`tail=$(repr(tail))` is invalid"))
    end
end

function StatsAPI.pvalue(dist::DiscreteUnivariateDistribution, x::Number; tail::Symbol=:both)
    _check_tail(tail)
    if tail === :both
        p = 2 * min(ccdf(dist, x-1), cdf(dist, x))
        min(p, oneunit(p)) # if P(X = x) > 0, then possibly p > 1
    elseif tail === :left
        cdf(dist, x)
    else # tail === :right
        ccdf(dist, x-1)
    end
end

function StatsAPI.pvalue(dist::ContinuousUnivariateDistribution, x::Number; tail::Symbol=:both)
    _check_tail(tail)
    if tail === :both
        p = 2 * min(cdf(dist, x), ccdf(dist, x))
        min(p, oneunit(p)) # if P(X = x) > 0, then possibly p > 1
    elseif tail === :left
        cdf(dist, x)
    else # tail === :right
        ccdf(dist, x)
    end
end
