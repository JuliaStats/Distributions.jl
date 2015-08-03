immutable NoncentralT <: ContinuousUnivariateDistribution
    ν::Float64
    λ::Float64

    function NoncentralT(ν::Real, λ::Real)
    	@check_args(NoncentralT, ν > zero(ν))
        @check_args(NoncentralT, λ >= zero(λ))
        new(ν, λ)
    end
end

@distr_support NoncentralT -Inf Inf

### Parameters

params(d::NoncentralT) = (d.ν, d.λ)


### Statistics

function mean(d::NoncentralT)
    if d.ν > 1.0
        isinf(d.ν) ? d.λ :
        sqrt(0.5*d.ν) * d.λ * gamma(0.5*(d.ν-1)) / gamma(0.5*d.ν)
    else
        NaN
    end
end

var(d::NoncentralT) = d.ν > 2.0 ? d.ν*(1+d.λ^2)/(d.ν-2.0) - mean(d)^2 : NaN


### Evaluation & Sampling

@_delegate_statsfuns NoncentralT ntdist ν λ

function rand(d::NoncentralT)
    z = randn()
    v = rand(Chisq(d.ν))
    (z+d.λ)/sqrt(v/d.ν)
end
