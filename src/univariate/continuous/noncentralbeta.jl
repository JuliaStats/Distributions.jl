immutable NoncentralBeta <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64
    λ::Float64
    
    function NoncentralBeta(α::Real, β::Real, λ::Real)
    	@check_args(NoncentralBeta, α > zero(α) && β > zero(β))
        @check_args(NoncentralBeta, λ >= zero(λ))
    	new(α, β, λ)
    end
end

@distr_support NoncentralBeta 0.0 1.0


### Parameters

params(d::NoncentralBeta) = (d.α, d.β, d.λ)


### Evaluation & Sampling

# TODO: add mean and var

@_delegate_statsfuns NoncentralBeta nbeta α β λ

function rand(d::NoncentralBeta)
    a = rand(NoncentralChisq(2.0 * d.α, d.β))
    b = rand(Chisq(2.0 * d.β))
    a / (a + b)
end
