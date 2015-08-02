immutable TDist <: ContinuousUnivariateDistribution
    ν::Float64

    function TDist(ν::Real)
    	ν > zero(ν) ||
            throw(ArgumentError("TDist: ν must be positive."))
        @compat new(Float64(ν))
    end
end

@distr_support TDist -Inf Inf


#### Parameters

dof(d::TDist) = d.ν
params(d::TDist) = (d.ν,)


#### Statistics

mean(d::TDist) = d.ν > 1.0 ? 0.0 : NaN
median(d::TDist) = 0.0
mode(d::TDist) = 0.0

function var(d::TDist)
    ν = d.ν
    ν > 2.0 ? ν / (ν - 2.0) :
    ν > 1.0 ? Inf : NaN
end

skewness(d::TDist) = d.ν > 3.0 ? 0.0 : NaN

function kurtosis(d::TDist)
    ν = d.ν
    ν > 4.0 ? 6.0 / (ν - 4.0) :
    ν > 2.0 ? Inf : NaN
end

function entropy(d::TDist)
    h = 0.5 * d.ν
    h1 = h + 0.5
    h1 * (digamma(h1) - digamma(h)) + 0.5 * log(d.ν) + lbeta(h, 0.5)
end


#### Evaluation & Sampling

@_delegate_statsfuns TDist tdist ν

rand(d::TDist) = StatsFuns.Rmath.tdistrand(d.ν)

function cf(d::TDist, t::Real)
    t == 0 && return complex(1.0)
    h = d.ν * 0.5
    q = d.ν * 0.25
    t2 = t*t
    complex(2*(q*t2)^q*besselk(h,sqrt(d.ν)*abs(t))/gamma(h))
end

gradlogpdf(d::TDist, x::Float64) = -((d.ν + 1.0) * x) / (x^2 + d.ν)
