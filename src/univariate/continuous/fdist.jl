immutable FDist <: ContinuousUnivariateDistribution
    d1::Float64
    d2::Float64

    function FDist(d1::Real, d2::Real)
        d1 > zero(d1) && d2 > zero(d2) || error("Degrees of freedom must be positive")
        @compat new(Float64(d1), Float64(d2))
    end
end

@_jl_dist_2p FDist f

@distr_support FDist 0.0 Inf


#### Parameters

params(d::FDist) = (d.d1, d.d2)


#### Statistics

mean(d::FDist) = (d2 = d.d2; d2 > 2.0 ? d2 / (d2 - 2.0) : NaN)

mode(d::FDist) = ((d1, d2) = params(d); d1 > 2.0 ? ((d1 - 2.0)/d1) * (d2 / (d2 + 2.0)) : 0.0)

function var(d::FDist)
    (d1, d2) = params(d)
    d2 > 4.0 ? 2.0 * d2^2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0)^2 * (d2 - 4.0)) : NaN
end

function skewness(d::FDist)
    (d1, d2) = params(d)
    if d2 > 6.0
        return (2.0 * d1 + d2 - 2.0) * sqrt(8.0 * (d2 - 4.0)) / ((d2 - 6.0) * sqrt(d1 * (d1 + d2 - 2.0)))
    else
        return NaN
    end
end

function kurtosis(d::FDist)
    (d1, d2) = params(d)
    if d2 > 8.0
        a = d1 * (5. * d2 - 22.) * (d1 + d2 - 2.) + (d2 - 4.) * (d2 - 2.)^2
        b = d1 * (d2 - 6.) * (d2 - 8.) * (d2 - 2.)
        return 12. * a / b
    else
        return NaN
    end
end

function entropy(d::FDist) 
    (d1, d2) = params(d)
    hd1 = d1 * 0.5
    hd2 = d2 * 0.5
    hs = (d1 + d2) * 0.5
    return log(d2 / d1) + lgamma(hd1) + lgamma(hd2) - lgamma(hs) +
        (1.0 - hd1) * digamma(hd1) + (-1.0 - hd2) * digamma(hd2) +
        hs * digamma(hs)
end
