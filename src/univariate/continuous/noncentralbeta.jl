immutable NoncentralBeta <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64
    λ::Float64
    function NoncentralBeta(α::Real, β::Real, λ::Real)
    	(α > 0.0 && β > 0.0) ||
            throw(ArgumentError("NoncentralBeta: α and β must be positive."))
        λ >= 0.0 ||
            throw(ArgumentError("NoncentralBeta: λ must be non-negative."))
    	@compat new(Float64(α), Float64(β), Float64(λ))
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
