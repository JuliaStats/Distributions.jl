immutable Beta <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64

    function Beta(a::Real, b::Real)
        (a > zero(a) && b > zero(b)) || error("α and β must be positive")
        @compat new(Float64(a), Float64(b))
    end

    Beta(α::Real) = Beta(α, α)
    Beta() = new(1.0, 1.0)
end

@_jl_dist_2p Beta beta

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

gradlogpdf(d::Beta, x::Float64) = 
    ((α, β) = params(d); 0.0 <= x <= 1.0 ? (α - 1.0) / x - (β - 1.0) / (1 - x) : 0.0)


## Fit model

# TODO: add MLE method (should be similar to Dirichlet)

# This is a moment-matching method (not MLE)
#
function fit(::Type{Beta}, x::AbstractArray)
    x_bar = mean(x)
    v_bar = varm(x, x_bar)
    α = x_bar * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    β = (1.0 - x_bar) * (((x_bar * (1.0 - x_bar)) / v_bar) - 1.0)
    Beta(α, β)
end


