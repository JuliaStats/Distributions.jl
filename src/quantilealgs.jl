# Various algorithms for computing quantile

function quantile_bisect(d::ContinuousUnivariateDistribution, p::Real,
                         lx::Real, rx::Real, tol::Real)

    # find quantile using bisect algorithm
    cl = cdf(d, lx)
    cr = cdf(d, rx)
    @assert cl <= p <= cr
    while rx - lx > tol
        m = (lx + rx)/2
        c = cdf(d, m)
        if p > c
            cl = c
            lx = m
        else
            cr = c
            rx = m
        end
    end
    return (lx + rx)/2
end

quantile_bisect(d::ContinuousUnivariateDistribution, p::Real) =
    quantile_bisect(d, p, minimum(d), maximum(d), 1.0e-12)

# if starting at mode, Newton is convergent for any unimodal continuous distribution, see:
#   Göknur Giner, Gordon K. Smyth (2014)
#   A Monotonically Convergent Newton Iteration for the Quantiles of any Unimodal
#   Distribution, with Application to the Inverse Gaussian Distribution
#   http://www.statsci.org/smyth/pubs/qinvgaussPreprint.pdf

function quantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12)
    T = promote_type(partype(d), typeof(p), typeof(xs))
    if 0 < p < 1
        x0 = T(xs)
        while true
            x = x0 + (p - cdf(d, x0)) / pdf(d, x0)
            abs(x-x0) >= max(abs(x),abs(x0))*tol || return x
            x0 = x
        end
    elseif p == 0
        return T(minimum(d))
    elseif p == 1
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function cquantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12)
    T = promote_type(partype(d), typeof(p), typeof(xs))
    if 0 < p < 1
        x0 = T(xs)
        while true
            x = x0 + (ccdf(d, x0)-p) / pdf(d, x0)
            abs(x-x0) >= max(abs(x),abs(x0))*tol || return x
            x0 = x
        end
    elseif p == 1
        return T(minimum(d))
    elseif p == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogcdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d), tol::Real=1e-12)
    T = promote_type(partype(d), typeof(lp), typeof(xs))
    if -Inf < lp < 0
        x0 = T(xs)
        if lp < logcdf(d,xs)
            while true
                x = x0 - exp(lp - logpdf(d,x0) + logexpm1(max(logcdf(d,x0)-lp,0)))
                abs(x-x0) >= max(abs(x),abs(x0))*tol || return x
                x0 = x
            end
        else
            while true
                x = x0 + exp(lp - logpdf(d,x0) + log1mexp(min(logcdf(d,x0)-lp,0)))
                abs(x-x0) >= max(abs(x),abs(x0))*tol || return x
                x0 = x
            end
        end
    elseif lp == -Inf
        return T(minimum(d))
    elseif lp == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogccdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d), tol::Real=1e-12)
    T = promote_type(partype(d), typeof(lp), typeof(xs))
    if -Inf < lp < 0
        x0 = T(xs)
        if lp < logccdf(d,xs)
            while true
                x = x0 + exp(lp - logpdf(d,x0) + logexpm1(max(logccdf(d,x0)-lp,0)))
                abs(x-x0) >= max(abs(x),abs(x0))*tol || return x
                x0 = x
            end
        else
            while true
                x = x0 - exp(lp - logpdf(d,x0) + log1mexp(min(logccdf(d,x0)-lp,0)))
                abs(x-x0) >= max(abs(x),abs(x0))*tol || return x
                x0 = x
            end
        end
    elseif lp == -Inf
        return T(maximum(d))
    elseif lp == 0
        return T(minimum(d))
    else
        return T(NaN)
    end
end

# A macro: specify that the quantile (and friends) of distribution D
# is computed using the newton method
macro quantile_newton(D)
    esc(quote
        quantile(d::$D, p::Real) = quantile_newton(d,p)
        cquantile(d::$D, p::Real) = cquantile_newton(d,p)
        invlogcdf(d::$D, lp::Real) = invlogcdf_newton(d,lp)
        invlogccdf(d::$D, lp::Real) = invlogccdf_newton(d,lp)
    end)
end
