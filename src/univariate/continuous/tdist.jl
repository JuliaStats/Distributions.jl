immutable TDist <: ContinuousUnivariateDistribution
    df::Float64
    function TDist(d::Real)
    	d > zero(d) || error("TDist: df must be positive")
        @compat new(Float64(d))
    end
end

@_jl_dist_1p TDist t

@distr_support TDist -Inf Inf


#### Parameters

dof(d::TDist) = d.df
params(d::TDist) = (d.df,)


#### Statistics

mean(d::TDist) = d.df > 1.0 ? 0.0 : NaN
median(d::TDist) = 0.0
mode(d::TDist) = 0.0

function var(d::TDist)
    df = d.df
    df > 2.0 ? df / (df - 2.0) :
    df > 1.0 ? Inf : NaN
end

skewness(d::TDist) = d.df > 3.0 ? 0.0 : NaN

function kurtosis(d::TDist)
    df = d.df
    df > 4.0 ? 6.0 / (df - 4.0) :
    df > 2.0 ? Inf : NaN
end

function entropy(d::TDist)
    hdf = 0.5 * d.df
    hdfph = hdf + 0.5
    hdfph * (digamma(hdfph) - digamma(hdf)) + 0.5 * log(d.df) + lbeta(hdf,0.5)
end


#### Evaluation
function cf(d::TDist, t::Real)
    t == 0 && return complex(1.0)
    h = d.df/2
    q = d.df/4
    t2 = t*t
    complex(2*(q*t2)^q*besselk(h,sqrt(d.df)*abs(t))/gamma(h))
end

gradlogpdf(d::TDist, x::Float64) = -((d.df + 1.0) * x) / (x^2 + d.df)

