# Various algorithms for computing quantile

function quantile_bisect(d::ContinuousUnivariateDistribution, p::Float64,
                         lx::Float64, rx::Float64, tol::Float64)

    # find quantile using bisect algorithm
    cl = cdf(d, lx)
    cr = cdf(d, rx)
    @assert cl <= p <= cr
    while rx - lx > tol
        m = 0.5 * (lx + rx)
        c = cdf(d, m)
        if p > c
            cl = c
            lx = m
        else
            cr = c
            rx = m
        end
    end
    return 0.5 * (lx + rx)
end

quantile_bisect(d::ContinuousUnivariateDistribution, p::Float64) =
    quantile_bisect(d, p, minimum(d), maximum(d), 1.0e-12)

# if starting at mode, Newton is convergent for any unimodal continuous distribution, see:
#   Göknur Giner, Gordon K. Smyth (2014)
#   A Monotonically Convergent Newton Iteration for the Quantiles of any Unimodal
#   Distribution, with Application to the Inverse Gaussian Distribution
#   http://www.statsci.org/smyth/pubs/qinvgaussPreprint.pdf

function quantile_newton(d::ContinuousUnivariateDistribution, p::Float64, xs::Float64=mode(d), tol::Float64=1e-12)
    if 0.0 < p < 1.0
        while true
            x = xs + (p - cdf(d, xs)) / pdf(d, xs)
            abs(x-xs) >= max(abs(x),abs(xs))*tol || return x
            xs = x
        end
    elseif p == 0.0
        return minimum(d)
    elseif p == 1.0
        return maximum(d)
    else
        return NaN
    end
end

function cquantile_newton(d::ContinuousUnivariateDistribution, p::Float64, xs::Float64=mode(d), tol::Float64=1e-12)
    if 0.0 < p < 1.0
        while true
            x = xs + (ccdf(d, xs)-p) / pdf(d, xs)
            abs(x-xs) >= max(abs(x),abs(xs))*tol || return x
            xs = x
        end
    elseif p == 1.0
        return minimum(d)
    elseif p == 0.0
        return maximum(d)
    else
        return NaN
    end
end

function invlogcdf_newton(d::ContinuousUnivariateDistribution, lp::Float64, xs::Float64=mode(d), tol::Float64=1e-12)
    if -Inf < lp < 0.0
        if lp < logcdf(d,xs)
            while true
                x = xs - exp(lp - logpdf(d,xs) + logexpm1(max(logcdf(d,xs)-lp,0.0)))
                abs(x-xs) >= max(abs(x),abs(xs))*tol || return x
                xs = x
            end
        else
            while true
                x = xs + exp(lp - logpdf(d,xs) + log1mexp(min(logcdf(d,xs)-lp,0.0)))
                abs(x-xs) >= max(abs(x),abs(xs))*tol || return x
                xs = x
            end
        end
    elseif lp == -Inf
        return minimum(d)
    elseif lp == 0.0
        return maximum(d)
    else
        return NaN
    end
end

function invlogccdf_newton(d::ContinuousUnivariateDistribution, lp::Float64, xs::Float64=mode(d), tol::Float64=1e-12)
    if -Inf < lp < 0.0
        if lp < logccdf(d,xs)
            while true
                x = xs + exp(lp - logpdf(d,xs) + logexpm1(max(logccdf(d,xs)-lp,0.0)))
                abs(x-xs) >= max(abs(x),abs(xs))*tol || return x
                xs = x
            end
        else
            while true
                x = xs - exp(lp - logpdf(d,xs) + log1mexp(min(logccdf(d,xs)-lp,0.0)))
                abs(x-xs) >= max(abs(x),abs(xs))*tol || return x
                xs = x
            end
        end
    elseif lp == -Inf
        return maximum(d)
    elseif lp == 0.0
        return minimum(d)
    else
        return NaN
    end
end

# A macro: specify that the quantile (and friends) of distribution D
# is computed using the newton method
macro quantile_newton(D)
    esc(quote
        quantile(d::$D, p::Float64) = quantile_newton(d,p)
        cquantile(d::$D, p::Float64) = cquantile_newton(d,p)
        invlogcdf(d::$D, lp::Float64) = invlogcdf_newton(d,lp)
        invlogccdf(d::$D, lp::Float64) = invlogccdf_newton(d,lp)
    end)
end
