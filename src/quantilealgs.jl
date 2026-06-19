# Various algorithms for computing quantile

function quantile_bisect(d::ContinuousUnivariateDistribution, p::Real, lx::T, rx::T) where {T<:Real}
    rx < lx && throw(ArgumentError("empty bracketing interval [$lx, $rx]"))

    # In some special cases, e.g. #1501, rx == lx`
    # If the distribution is degenerate the check below can fail, hence we skip it
    if rx == lx
        # Returns `lx` of the same type as `(lx + rx) / 2`
        # For specific types such as `Float64` it is more performant than `oftype((lx + rx) / 2, lx)`
        return middle(lx)
    end

    # Rely on Roots' default tolerances for the bracketing solver. Unlike the previous
    # hand-rolled bisection, ITP uses a relative `xrtol = eps`, so it converges even for
    # brackets far from zero where an absolute tolerance is below the floating-point spacing
    # (#1611).
    return find_zero(x -> cdf(d, x) - p, (lx, rx), ITP())
end

function quantile_bisect(d::ContinuousUnivariateDistribution, p::Real, lx::Real, rx::Real)
    return quantile_bisect(d, p, promote(lx, rx)...)
end

quantile_bisect(d::ContinuousUnivariateDistribution, p::Real) =
    quantile_bisect(d, p, minimum(d), maximum(d))

# if starting at mode, Newton is convergent for any unimodal continuous distribution, see:
#   Göknur Giner, Gordon K. Smyth (2014)
#   A Monotonically Convergent Newton Iteration for the Quantiles of any Unimodal
#   Distribution, with Application to the Inverse Gaussian Distribution
#   http://www.statsci.org/smyth/pubs/qinvgaussPreprint.pdf

function quantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d))
    x = xs + (p - cdf(d, xs)) / pdf(d, xs)
    T = typeof(x)
    F(x) = cdf(d, x) - p
    f(x) = pdf(d, x)
    if 0 < p < 1
        # Newton's method with an ITP bracketing fallback: Roots switches to ITP whenever the
        # iteration brackets the root (sign change of `F`), which resolves the oscillation/stalling
        # observed for extreme quantiles (#2061, #1898). Roots' default tolerances are used.
        return find_zero((F, f), x, Newton(), ITP())
    elseif p == 0
        return T(minimum(d))
    elseif p == 1
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function cquantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d))
    x = xs + (ccdf(d, xs)-p) / pdf(d, xs)
    T = typeof(x)
    F(x) = ccdf(d, x) - p
    f(x) = -pdf(d, x)
    if 0 < p < 1
        return find_zero((F, f), x, Newton(), ITP())
    elseif p == 1
        return T(minimum(d))
    elseif p == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogcdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d))
    T = typeof(lp - logpdf(d,xs))
    F(x) = logcdf(d, x) - lp
    f(x) = exp(logpdf(d, x) - logcdf(d, x))
    if -Inf < lp < 0
        x0 = T(xs)
        x = lp < logcdf(d,x0) ?
            x0 - exp(lp - logpdf(d,x0) + logexpm1(max(logcdf(d,x0)-lp,0))) :
            x0 + exp(lp - logpdf(d,x0) + log1mexp(min(logcdf(d,x0)-lp,0)))
        return find_zero((F, f), x, Newton(), ITP())
    elseif lp == -Inf
        return T(minimum(d))
    elseif lp == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogccdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d))
    T = typeof(lp - logpdf(d,xs))
    F(x) = logccdf(d, x) - lp
    f(x) = -exp(logpdf(d, x) - logccdf(d, x))
    if -Inf < lp < 0
        x0 = T(xs)
        x = lp < logccdf(d,x0) ?
            x0 + exp(lp - logpdf(d,x0) + logexpm1(max(logccdf(d,x0)-lp,0))) :
            x0 - exp(lp - logpdf(d,x0) + log1mexp(min(logccdf(d,x0)-lp,0)))
        return find_zero((F, f), x, Newton(), ITP())
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
