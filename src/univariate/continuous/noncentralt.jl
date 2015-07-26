immutable NoncentralT <: ContinuousUnivariateDistribution
    df::Float64
    λ::Float64

    function NoncentralT(d::Real, nc::Real)
    	d >= zero(d) && nc >= zero(nc) || error("df and λ must be non-negative")
        @compat new(Float64(d), Float64(nc))
    end
end

@distr_support NoncentralT -Inf Inf

function mean(d::NoncentralT)
    if d.df > 1.0
        if isinf(d.df)
            d.λ
        else
            sqrt(0.5*d.df)*d.λ*gamma(0.5*(d.df-1))/gamma(0.5*d.df)
        end
    else
        NaN
    end
end

var(d::NoncentralT) = d.df > 2.0 ? d.df*(1+d.λ^2)/(d.df-2.0) - mean(d)^2 : NaN

@_delegate_statsfuns NoncentralT ntdist df λ

function rand(d::NoncentralT)
    z = randn()
    v = rand(Chisq(d.df))
    (z+d.λ)/sqrt(v/d.df)
end
