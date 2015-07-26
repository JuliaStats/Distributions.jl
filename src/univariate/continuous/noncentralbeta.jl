immutable NoncentralBeta <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64
    λ::Float64
    function NoncentralBeta(a::Real, b::Real, nc::Real)
    	a > 0.0 && b > 0.0 && nc >= 0.0 ||
            error("alpha and beta must be > 0 and ncp >= 0")
    	@compat new(Float64(a), Float64(b), Float64(nc))
    end
end

@distr_support NoncentralBeta 0.0 1.0

# TODO: add mean and var

@_delegate_statsfuns NoncentralBeta nbeta α β λ

function rand(d::NoncentralBeta)
    a = rand(NoncentralChisq(2.0 * d.α, d.β))
    b = rand(Chisq(2.0 * d.β))
    a / (a+b)
end
