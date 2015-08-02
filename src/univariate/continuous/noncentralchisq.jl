immutable NoncentralChisq <: ContinuousUnivariateDistribution
    ν::Float64
    λ::Float64
    function NoncentralChisq(ν::Real, λ::Real)
        ν > zero(ν) ||
            throw(ArgumentError("NoncentralChisq: ν must be positive."))
    	λ >= zero(λ) ||
            error("NoncentralChisq: λ must be non-negative.")
    	@compat new(Float64(ν), Float64(λ))
    end
end

@distr_support NoncentralChisq 0.0 Inf

### Parameters

params(d::NoncentralChisq) = (d.ν, d.λ)


### Statistics

mean(d::NoncentralChisq) = d.ν + d.λ
var(d::NoncentralChisq) = 2.0*(d.ν + 2.0*d.λ)
skewness(d::NoncentralChisq) = 2.0*sqrt2*(d.ν + 3.0*d.λ)/sqrt(d.ν + 2.0*d.λ)^3
kurtosis(d::NoncentralChisq) = 12.0*(d.ν + 4.0*d.λ)/(d.ν + 2.0*d.λ)^2

function mgf(d::NoncentralChisq, t::Float64)
    exp(d.λ * t/(1.0 - 2.0 * t))*(1.0 - 2.0 * t)^(-d.ν / 2.0)
end

function cf(d::NoncentralChisq, t::Float64)
    cis(d.λ * t/(1.0 - 2.0 * im * t))*(1.0 - 2.0 * im * t)^(-d.ν / 2.0)
end


### Evaluation & Sampling

@_delegate_statsfuns NoncentralChisq nchisq ν λ

rand(d::NoncentralChisq) = StatsFuns.Rmath.nchisqrand(d.ν, d.λ)
