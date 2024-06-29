# Moments of the truncated log-normal can be computed directly from the moment generating
# function of the truncated normal:
#   Let Y ~ LogNormal(μ, σ) truncated to (a, b). Then log(Y) ~ Normal(μ, σ) truncated
#   to (log(a), log(b)), and E[Y^n] = E[(e^log(Y))^n] = E[e^(nlog(Y))] = mgf(log(Y), n).

# Given `truncate(LogNormal(μ, σ), a, b)`, return `truncate(Normal(μ, σ), log(a), log(b))`
function _truncnorm(d::Truncated{<:LogNormal})
    μ, σ = params(d.untruncated)
    T = float(partype(d))
    a = d.lower === nothing || d.lower <= 0 ? nothing : log(T(d.lower))
    b = d.upper === nothing || isinf(d.upper) ? nothing : log(T(d.upper))
    return truncated(Normal{T}(T(μ), T(σ)), a, b)
end

mean(d::Truncated{<:LogNormal}) = mgf(_truncnorm(d), 1)

function var(d::Truncated{<:LogNormal})
    tn = _truncnorm(d)
    # Ensure the variance doesn't end up negative, which can occur due to numerical issues
    return max(mgf(tn, 2) - mgf(tn, 1)^2, 0)
end

function skewness(d::Truncated{<:LogNormal})
    tn = _truncnorm(d)
    m1 = mgf(tn, 1)
    m2 = mgf(tn, 2)
    m3 = mgf(tn, 3)
    sqm1 = m1^2
    v = m2 - sqm1
    return (m3 + m1 * (-3 * m2 + 2 * sqm1)) / (v * sqrt(v))
end

function kurtosis(d::Truncated{<:LogNormal})
    tn = _truncnorm(d)
    m1 = mgf(tn, 1)
    m2 = mgf(tn, 2)
    m3 = mgf(tn, 3)
    m4 = mgf(tn, 4)
    v = m2 - m1^2
    return @horner(m1, m4, -4m3, 6m2, 0, -3) / v^2 - 3
end

# TODO: The entropy can be written "directly" as well, according to Mathematica, but
# the expression for it fills me with regret. There are some recognizable components,
# so a sufficiently motivated person could try to manually simplify it into something
# comprehensible. For reference, you can obtain the entropy with Mathematica like so:
#
#   d = TruncatedDistribution[{a, b}, LogNormalDistribution[m, s]];
#   Expectation[-LogLikelihood[d, {x}], Distributed[x, d],
#               Assumptions -> Element[x | m | s | a | b, Reals] && s > 0 && 0 < a < x < b]
