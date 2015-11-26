immutable Beta <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64

    function Beta(α::Real, β::Real)
        @check_args(Beta, α > zero(α) && β > zero(β))
        new(α, β)
    end
    Beta(α::Real) = Beta(α, α)
    Beta() = new(1.0, 1.0)
end

@distr_support Beta 0.0 1.0

#### Parameters

params(d::Beta) = (d.α, d.β)


#### Statistics

mean(d::Beta) = ((α, β) = params(d); α / (α + β))

function mode(d::Beta)
    (α, β) = params(d)
    (α > 1.0 && β > 1.0) || error("mode is defined only when α > 1 and β > 1.")
    return (α - 1.0) / (α + β - 2.0)
end

modes(d::Beta) = [mode(d)]

function var(d::Beta)
    (α, β) = params(d)
    s = α + β
    return (α * β) / (abs2(s) * (s + 1.0))
end

meanlogx(d::Beta) = ((α, β) = params(d); digamma(α) - digamma(α + β))

varlogx(d::Beta) = ((α, β) = params(d); trigamma(α) - trigamma(α + β))
stdlogx(d::Beta) = sqrt(varlogx(d))

function skewness(d::Beta)
    (α, β) = params(d)
    if α == β
        return 0.0
    else
        s = α + β
        (2.0 * (β - α) * sqrt(s + 1.0)) / ((s + 2.0) * sqrt(α * β))
    end
end

function kurtosis(d::Beta)
    α, β = params(d)
    s = α + β
    p = α * β
    6.0 * (abs2(α - β) * (s + 1.0) - p * (s + 2.0)) / (p * (s + 2.0) * (s + 3.0))
end

function entropy(d::Beta)
    α, β = params(d)
    s = α + β
    lbeta(α, β) - (α - 1.0) * digamma(α) - (β - 1.0) * digamma(β) +
        (s - 2.0) * digamma(s)
end


#### Evaluation

@_delegate_statsfuns Beta beta α β

gradlogpdf(d::Beta, x::Float64) =
    ((α, β) = params(d); 0.0 <= x <= 1.0 ? (α - 1.0) / x - (β - 1.0) / (1 - x) : 0.0)


#### Sampling

rand(d::Beta) = StatsFuns.Rmath.betarand(d.α, d.β)


#### Fit model

# This is the MLE method 
function fit_mle{T<:Real}(::Type{Beta}, data::AbstractArray{T})
    initDist = fit(Beta, data)
    s1 = sum(log(data))
    s2 = sum(log(1-data))
    function f!(x, fvec)
        fvec[1] = s1 - length(data)*(digamma(x[1])-digamma(x[1]+x[2]))
        fvec[2] = s2 - length(data)*(digamma(x[2])-digamma(x[1]+x[2]))
    end
    sol = nlsolve(f!, [initDist.α, initDist.β])
    Beta(sol.zero[1], sol.zero[2])
end

# This is a moment-matching method (not MLE)
#
function fit{T<:Real}(::Type{Beta}, x::AbstractArray{T})
    x_bar = mean(x)
    v_bar = varm(x, x_bar)
    α = x_bar * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    β = (1.0 - x_bar) * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    Beta(α, β)
end
