immutable Erlang <: ContinuousUnivariateDistribution
    α::Int
    θ::Float64

    function Erlang(α::Real, θ::Real)
        @check_args(Erlang, isinteger(α) && α >= zero(α))
        new(α, θ)
    end
    Erlang(α::Real) = Erlang(α, 1.0)
    Erlang() = new(1, 1.0)
end

@distr_support Erlang 0.0 Inf

#### Parameters

shape(d::Erlang) = d.α
scale(d::Erlang) = d.θ
rate(d::Erlang) = inv(d.θ)
params(d::Erlang) = (d.α, d.θ)

#### Statistics

mean(d::Erlang) = d.α * d.θ
var(d::Erlang) = d.α * d.θ^2
skewness(d::Erlang) = 2.0 / sqrt(d.α)
kurtosis(d::Erlang) = 6.0 / d.α

function mode(d::Erlang)
    (α, θ) = params(d)
    α >= 1.0 ? θ * (α - 1.0) : error("Erlang has no mode when α < 1.0")
end

function entropy(d::Erlang)
    (α, θ) = params(d)
    α + lgamma(α) + (1.0 - α) * digamma(α) + log(θ)
end

mgf(d::Erlang, t::Real) = (1.0 - t * d.θ)^(-d.α)
cf(d::Erlang, t::Real)  = (1.0 - im * t * d.θ)^(-d.α)


#### Evaluation & Sampling

@_delegate_statsfuns Erlang gamma α θ

rand(d::Erlang) = StatsFuns.Rmath.gammarand(d.α, d.θ)
