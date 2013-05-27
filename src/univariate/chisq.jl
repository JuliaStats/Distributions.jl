immutable Chisq <: ContinuousUnivariateDistribution
    df::Float64 # non-integer degrees of freedom are meaningful
    function Chisq(d::Real)
        if d > 0.0
            new(float64(d))
        else
            error("df must be positive")
        end
    end
end

@_jl_dist_1p Chisq chisq

function entropy(d::Chisq)
    x = d.df / 2.0 + log(2.0) + lgamma(d.df / 2.0)
    x += (1.0 - d.df / 2.0) * digamma(d.df / 2.0)
    return x
end

insupport(d::Chisq, x::Number) = isreal(x) && isfinite(x) && 0.0 <= x

kurtosis(d::Chisq) = 12.0 / d.df

mean(d::Chisq) = d.df

# rand - the distribution chi^2(df) is 2 * gamma(df / 2)
# for integer n, a chi^2(n) is the sum of n squared standard normals
function rand(d::Chisq)
    if d.df == 1
        return randn()^2
    else
        return 2.0 * rand(Gamma(d.df / 2.0))
    end
end

function rand!(d::Chisq, A::Array{Float64})
    if d.df == 1
        for i in 1:length(A)
            A[i] = randn()^2
        end
        return A
    end
    if d.df >= 2
        dpar = d.df / 2.0 - 1.0 / 3.0
    else
        error("require degrees of freedom df >= 2")
    end
    cpar = 1.0 / sqrt(9.0 * dpar)
    for i in 1:length(A)
        A[i] = 2.0 * randg2(dpar, cpar)
    end
    A
end

skewness(d::Chisq) = sqrt(8.0 / d.df)

var(d::Chisq) = 2.0 * d.df
