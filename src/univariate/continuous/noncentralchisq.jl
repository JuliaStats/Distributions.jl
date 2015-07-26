immutable NoncentralChisq <: ContinuousUnivariateDistribution
    df::Float64
    λ::Float64
    function NoncentralChisq(d::Real, nc::Real)
    	d >= zero(d) && nc >= zero(nc) || error("df and ncp must be non-negative")
    	@compat new(Float64(d), Float64(nc))
    end
end

@distr_support NoncentralChisq 0.0 Inf

mean(d::NoncentralChisq) = d.df + d.λ
var(d::NoncentralChisq) = 2.0*(d.df + 2.0*d.λ)
skewness(d::NoncentralChisq) = 2.0*sqrt2*(d.df + 3.0*d.λ)/sqrt(d.df + 2.0*d.λ)^3
kurtosis(d::NoncentralChisq) = 12.0*(d.df + 4.0*d.λ)/(d.df + 2.0*d.λ)^2

function mgf(d::NoncentralChisq, t::Real)
    k = d.df
    exp(d.λ * t/(1.0 - 2.0 * t))*(1.0 - 2.0 * t)^(-k / 2.0)
end

function cf(d::NoncentralChisq, t::Real)
    cis(d.λ * t/(1.0 - 2.0 * im * t))*(1.0 - 2.0 * im * t)^(-d.df / 2.0)
end

@_delegate_statsfuns NoncentralChisq nchisq df λ

rand(d::NoncentralChisq) = StatsFuns.Rmath.nchisqrand(d.df, d.λ)
