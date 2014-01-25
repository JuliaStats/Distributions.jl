immutable Beta <: ContinuousUnivariateDistribution
    alpha::Float64
    beta::Float64
    function Beta(a::Real, b::Real)
        (a > zero(a) && b > zero(b)) || error("alpha and beta must be positive")
        new(float64(a), float64(b))
    end
end

### handling support

@continuous_distr_support Beta 0.0 1.0

Beta(a::Real) = Beta(a, a) # symmetric in [0, 1]
Beta() = Beta(1.0) # uniform

mean(d::Beta) = d.alpha / (d.alpha + d.beta)

median(d::Beta) = quantile(d, 0.5)

function mode(d::Beta)
    α, β = d.alpha, d.beta
    α > 1.0 && β > 1.0 || error("Beta with α <= 1 or β <= 1 has no modes")
    (α - 1.0) / (α + β - 2.0)
end

modes(d::Beta) = [mode(d)]

function var(d::Beta)
    ab = d.alpha + d.beta
    d.alpha * d.beta / (ab * ab * (ab + 1.0))
end

function skewness(d::Beta)
    num = 2.0 * (d.beta - d.alpha) * sqrt(d.alpha + d.beta + 1.0)
    den = (d.alpha + d.beta + 2.0) * sqrt(d.alpha * d.beta)
    num / den
end
function kurtosis(d::Beta)
    α, β = d.alpha, d.beta
    num = 6.0 * ((α - β)^2 * (α + β + 1.0) - α * β * (α + β + 2.0))
    den = α * β * (α + β + 2.0) * (α + β + 3.0)
    num / den
end

function entropy(d::Beta)
    o = lbeta(d.alpha, d.beta)
    o -= (d.alpha - 1.0) * digamma(d.alpha)
    o -= (d.beta - 1.0) * digamma(d.beta)
    o += (d.alpha + d.beta - 2.0) * digamma(d.alpha + d.beta)
    o
end


pdf(d::Beta, x::Real) = insupport(d, x) ? brcomp(d.alpha, d.beta, x, 1.0 - x)/(x*(1.0 - x)) : 0.0


function cdf(d::Beta, x::Real)
    if x >= 1 return 1.0 end
    if x <= 0 return 0.0 end
    return bratio(d.alpha, d.beta, x)
end

function ccdf(d::Beta, x::Real)
    if x >= 1 return 0.0 end
    if x <= 0 return 1.0 end
    return bratio(d.beta, d.alpha, 1.0-x)
end



function rand(d::Beta)
    u = randg(d.alpha)
    u / (u + randg(d.beta))
end

function rand!(d::Beta, A::Array{Float64})
    α = d.alpha
    β = d.beta

    da = (α <= 1.0 ? α + 1.0 : α) - 1.0 / 3.0
    ca = 1.0 / sqrt(9.0 * da)

    db = (β <= 1.0 ? β + 1.0 : β) - 1.0 / 3.0
    cb = 1.0 / sqrt(9.0 * db)

    for i = 1:length(A)
        u = randg2(da, ca)
        A[i] = u / (u + randg2(db, cb))
    end
    A
end

## Fit model

# TODO: add MLE method (should be similar to Dirichlet)

# This is a moment-matching method (not MLE)
#
function fit(::Type{Beta}, x::Array)
    for xi in x
        insupport(Beta, xi) || error("Beta observations must be in [0,1]")
    end
    x_bar = mean(x)
    v_bar = varm(x, x_bar)
    α = x_bar * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    β = (1.0 - x_bar) * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    Beta(α, β)
end


