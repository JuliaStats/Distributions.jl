# Kolmogorov distribution
# defined as the sup_{t \in [0,1]} |B(t)|, where B(t) is a Brownian bridge
# used in the Kolmogorov--Smirnov test for large n.
immutable Kolmogorov <: ContinuousUnivariateDistribution
end

insupport(::Kolmogorov, x::Real) = zero(x) <= x < Inf
insupport(::Type{Kolmogorov}, x::Real) = zero(x) <= x < Inf

mean(d::Kolmogorov) = sqrt(pi/2.0) * log(2.0)
var(d::Kolmogorov) = pi^2/12.0 - pi/2.0*log(2.0)^2
# TODO: higher-order moments also exist, can be obtained by differentiating series

# cdf and ccdf are based on series truncation.
# two different series are available, e.g. see:
#   N. Smirnov, "Table for Estimating the Goodness of Fit of Empirical Distributions",
#   The Annals of Mathematical Statistics , Vol. 19, No. 2 (Jun., 1948), pp. 279-281
#   http://projecteuclid.org/euclid.aoms/1177730256
# use one series for small x, one for large x
# 5 terms seems to be sufficient for Float64 accuracy
# some divergence from Smirnov's table in 6th decimal near 1.0 (e.g. 1.04): occurs in 
# both series so assume error in table.

function cdf(d::Kolmogorov,x::Real)
    if x <= 0.0
        return 0.0
    elseif x > 1.0
        return 1.0-ccdf(d,x)
    end
    s = 0.0
    for i = 1:5
        s += exp(-((2*i-1)*π/x)^2/8.0)
    end
    √2π*s/x
end

function ccdf(d::Kolmogorov,x::Real)
    if x <= 1.0
        return 1.0-cdf(d,x)
    end
    s = 0.0
    for i = 1:5
        s += (iseven(i) ? -1 : 1)*exp(-2.0*(i*x)^2)
    end
    2.0*s
end
