immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom are meaningful
    function Chisq(d::Real)
        d > zero(d) || error("df must be positive")
        new(float64(d))
    end
end

@_jl_dist_1p Chisq chisq

convert(::Type{Gamma},d::Chisq) = Gamma(0.5*d.df,2.0)

## Support
@continuous_distr_support Chisq 0.0 Inf

## Properties
mean(d::Chisq) = d.df
mode(d::Chisq) = d.df > 2.0 ? d.df - 2.0 : 0.0

var(d::Chisq) = 2.0 * d.df
skewness(d::Chisq) = sqrt(8.0 / d.df)
kurtosis(d::Chisq) = 12.0 / d.df

function entropy(d::Chisq) 
    hdf = 0.5*d.df
    hdf + logtwo + lgamma(hdf) + (1.0-hdf)*digamma(hdf)
end

## Functions
pdf(d::Chisq, x::Real) = pdf(convert(Gamma,d),x)
logpdf(d::Chisq, x::Real) = logpdf(convert(Gamma,d),x)

gradloglik(d::Chisq, x::Float64) = x > zero(x) ? (0.5*d.df-1.0)/x - 0.5 : 0.0

mgf(d::Chisq, t::Real) = (1.0 - 2.0*t)^(-0.5*d.df)
cf(d::Chisq, t::Real) = (1.0 - 2.0*im*t)^(-0.5*d.df)


## Sampling
# for integer n, a chi^2(n) is the sum of n squared standard normals
function rand(d::Chisq)
    d.df == 1 ? randn()^2 : rand(convert(Gamma,d))
end

function rand!(d::Chisq, A::Array{Float64})
    if d.df == 1
        for i = 1:length(A)
            @inbounds A[i] = randn()^2
        end
    else
        s = GammaSampler(d.df / 2.0)
        for i = 1:length(A)
            @inbounds A[i] = 2.0 * rand(s)
        end
    end
    return A
end

