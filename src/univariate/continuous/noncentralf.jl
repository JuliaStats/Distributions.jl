immutable NoncentralF <: ContinuousUnivariateDistribution
    ν1::Float64
    ν2::Float64
    λ::Float64

    function NoncentralF(ν1::Real, ν2::Real, λ::Real)
        (ν1 > zero(ν1) && ν2 > zero(ν2)) ||
            throw(ArgumentError("NoncentralF: ν1 and ν2 must be both positive."))
        λ > zero(λ) ||
            throw(ArgumentError("NoncentralF: λ must be non-negative."))
	    @compat new(Float64(ν1), Float64(ν2), Float64(λ))
    end
end

@distr_support NoncentralF 0.0 Inf


### Parameters

params(d::NoncentralF) = (d.ν1, d.ν2, d.λ)


### Statistics

mean(d::NoncentralF) = d.ν2 > 2.0 ? d.ν2 / (d.ν2 - 2.0) * (d.ν1 + d.λ) / d.ν1 : NaN

var(d::NoncentralF) = d.ν2 > 4.0 ? 2.0 * d.ν2^2 *
		       ((d.ν1+d.λ)^2 + (d.ν2 - 2.0)*(d.ν1 + 2.0*d.λ)) /
		       (d.ν1 * (d.ν2 - 2.0)^2 * (d.ν2 - 4.0)) : NaN


### Evaluation & Sampling

@_delegate_statsfuns NoncentralF nfdist ν1 ν2 λ

function rand(d::NoncentralF)
    r1 = rand(NoncentralChisq(d.ν1,d.λ)) / d.ν1
    r2 = rand(Chisq(d.ν2)) / d.ν2
    r1 / r2
end
