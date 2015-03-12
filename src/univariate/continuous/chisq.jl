immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64

    function Chisq(k::Real)
        k > zero(k) || error("The degree of freedom k must be positive")
        @compat new(Float64(k))
    end
end

@_jl_dist_1p Chisq chisq

@distr_support Chisq 0.0 Inf


#### Parameters

dof(d::Chisq) = d.df
params(d::Chisq) = (d.df,)


#### Statistics

mean(d::Chisq) = dof(d)

var(d::Chisq) = 2.0 * dof(d)

skewness(d::Chisq) = sqrt(8.0 / dof(d))

kurtosis(d::Chisq) = 12.0 / dof(d)

mode(d::Chisq) = d.df > 2.0 ? d.df - 2.0 : 0.0

function median(d::Chisq; approx::Bool=false)
    if approx
        k = dof(d)
        return k * (1.0 - 2.0 / (9.0 * k))^3
    else
        return quantile(d, 0.5)
    end
end

function entropy(d::Chisq)
    hk = 0.5 * dof(d)
    hk + logtwo + lgamma(hk) + (1.0 - hk) * digamma(hk)
end


#### Evaluation

mgf(d::Chisq, t::Real) = (1.0 - 2.0 * t)^(-dof(d) / 2.0)

cf(d::Chisq, t::Real) = (1.0 - 2.0 * im * t)^(-dof(d) / 2.0)

gradlogpdf(d::Chisq, x::Float64) =  x >= 0.0 ? (dof(d) / 2.0 - 1) / x - 0.5 : 0.0

