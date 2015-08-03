immutable FDist <: ContinuousUnivariateDistribution
    ν1::Float64
    ν2::Float64

    function FDist(ν1::Real, ν2::Real)
        @check_args(FDist, ν1 > zero(ν1) && ν2 > zero(ν2))
        new(ν1, ν2)
    end
end

@distr_support FDist 0.0 Inf


#### Parameters

params(d::FDist) = (d.ν1, d.ν2)


#### Statistics

mean(d::FDist) = (ν2 = d.ν2; ν2 > 2.0 ? ν2 / (ν2 - 2.0) : NaN)

mode(d::FDist) = ((ν1, ν2) = params(d); ν1 > 2.0 ? ((ν1 - 2.0)/ν1) * (ν2 / (ν2 + 2.0)) : 0.0)

function var(d::FDist)
    (ν1, ν2) = params(d)
    ν2 > 4.0 ? 2.0 * ν2^2 * (ν1 + ν2 - 2.0) / (ν1 * (ν2 - 2.0)^2 * (ν2 - 4.0)) : NaN
end

function skewness(d::FDist)
    (ν1, ν2) = params(d)
    if ν2 > 6.0
        return (2.0 * ν1 + ν2 - 2.0) * sqrt(8.0 * (ν2 - 4.0)) / ((ν2 - 6.0) * sqrt(ν1 * (ν1 + ν2 - 2.0)))
    else
        return NaN
    end
end

function kurtosis(d::FDist)
    (ν1, ν2) = params(d)
    if ν2 > 8.0
        a = ν1 * (5. * ν2 - 22.) * (ν1 + ν2 - 2.) + (ν2 - 4.) * (ν2 - 2.)^2
        b = ν1 * (ν2 - 6.) * (ν2 - 8.) * (ν2 - 2.)
        return 12. * a / b
    else
        return NaN
    end
end

function entropy(d::FDist)
    (ν1, ν2) = params(d)
    hν1 = ν1 * 0.5
    hν2 = ν2 * 0.5
    hs = (ν1 + ν2) * 0.5
    return log(ν2 / ν1) + lgamma(hν1) + lgamma(hν2) - lgamma(hs) +
        (1.0 - hν1) * digamma(hν1) + (-1.0 - hν2) * digamma(hν2) +
        hs * digamma(hs)
end

#### Evaluation & Sampling

@_delegate_statsfuns FDist fdist ν1 ν2

rand(d::FDist) = StatsFuns.Rmath.fdistrand(d.ν1, d.ν2)
