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

    # base tolerance on types to support e.g. `Float32` (avoids an infinite loop)
    # ≈ 3.7e-11 for Float64
    # ≈ 2.4e-5 for Float32
    tol = cbrt(eps(float(T)))^2
    # find quantile using bisect algorithm
    cl = cdf(d, lx)
    cr = cdf(d, rx)
    cl <= p <= cr ||
        throw(ArgumentError("[$lx, $rx] is not a valid bracketing interval for `quantile(d, $p)`"))
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

function quantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12, max_iter::Int=150)
    x = xs + (p - cdf(d, xs)) / pdf(d, xs)
    T = typeof(x)
    if 0 < p < 1
        x0 = T(xs)
        x2 = x0
        have_x2 = false
        tol = max(tol, cbrt(eps(float(T)))^2)
        for _ in 1:max_iter
            abs(x-x0) <= max(abs(x),abs(x0)) * tol && return x
            if have_x2 && abs(x-x2) <= max(abs(x),abs(x2)) * cbrt(eps(float(T)))^2
                @warn "quantile_newton: 2-cycle detected, falling back to bisection" p=p maxlog=1
                return quantile_bisect(d, p, minmax(x, x0)...)
            end
            x2 = x0
            have_x2 = true
            x0 = x
            x = x0 + (p - cdf(d, x0)) / pdf(d, x0)
        end
        @warn "quantile_newton: maximum iterations reached, falling back to bisection" p=p maxlog=1
        return quantile_bisect(d, p, minmax(x, x0)...)
    elseif p == 0
        return T(minimum(d))
    elseif p == 1
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function cquantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12, max_iter::Int=150)
    x = xs + (ccdf(d, xs)-p) / pdf(d, xs)
    T = typeof(x)
    if 0 < p < 1
        x0 = T(xs)
        x2 = x0
        have_x2 = false
        tol = max(tol, cbrt(eps(float(T)))^2)
        for _ in 1:max_iter
            abs(x-x0) <= max(abs(x),abs(x0)) * tol && return x
            if have_x2 && abs(x-x2) <= max(abs(x),abs(x2)) * cbrt(eps(float(T)))^2
                @warn "cquantile_newton: 2-cycle detected, falling back to bisection" p=p maxlog=1
                return quantile_bisect(d, 1 - p, minmax(x, x0)...)
            end
            x2 = x0
            have_x2 = true
            x0 = x
            x = x0 + (ccdf(d, x0)-p) / pdf(d, x0)
        end
        @warn "cquantile_newton: maximum iterations reached, falling back to bisection" p=p maxlog=1
        return quantile_bisect(d, 1 - p, minmax(x, x0)...)
    elseif p == 1
        return T(minimum(d))
    elseif p == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogcdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d), tol::Real=1e-12, max_iter::Int=150)
    T = typeof(lp - logpdf(d,xs))
    if -Inf < lp < 0
        x0 = T(xs)
        x2 = x0
        have_x2 = false
        tol = max(tol, cbrt(eps(float(T)))^2)
        if lp < logcdf(d,x0)
            x = x0 - exp(lp - logpdf(d,x0) + logexpm1(max(logcdf(d,x0)-lp,0)))
            for _ in 1:max_iter
                abs(x-x0) <= max(abs(x),abs(x0)) * tol && return x
                if have_x2 && abs(x-x2) <= max(abs(x),abs(x2)) * cbrt(eps(float(T)))^2
                    @warn "invlogcdf_newton: 2-cycle detected, falling back to bisection" lp=lp maxlog=1
                    return quantile_bisect(d, exp(lp), minmax(x, x0)...)
                end
                x2 = x0
                have_x2 = true
                x0 = x
                x = x0 - exp(lp - logpdf(d,x0) + logexpm1(max(logcdf(d,x0)-lp,0)))
            end
        else
            x = x0 + exp(lp - logpdf(d,x0) + log1mexp(min(logcdf(d,x0)-lp,0)))
            for _ in 1:max_iter
                abs(x-x0) <= max(abs(x),abs(x0))*tol && return x
                if have_x2 && abs(x-x2) <= max(abs(x),abs(x2)) * cbrt(eps(float(T)))^2
                    @warn "invlogcdf_newton: 2-cycle detected, falling back to bisection" lp=lp maxlog=1
                    return quantile_bisect(d, exp(lp), minmax(x, x0)...)
                end
                x2 = x0
                have_x2 = true
                x0 = x
                x = x0 + exp(lp - logpdf(d,x0) + log1mexp(min(logcdf(d,x0)-lp,0)))
            end
        end
        @warn "invlogcdf_newton: maximum iterations reached, falling back to bisection" lp=lp maxlog=1
        return quantile_bisect(d, exp(lp), minmax(x, x0)...)
    elseif lp == -Inf
        return T(minimum(d))
    elseif lp == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogccdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d), tol::Real=1e-12, max_iter::Int=150)
    T = typeof(lp - logpdf(d,xs))
    if -Inf < lp < 0
        x0 = T(xs)
        x2 = x0
        have_x2 = false
        tol = max(tol, cbrt(eps(float(T)))^2)
        if lp < logccdf(d,x0)
            x = x0 + exp(lp - logpdf(d,x0) + logexpm1(max(logccdf(d,x0)-lp,0)))
            for _ in 1:max_iter
                abs(x-x0) <= max(abs(x),abs(x0)) * tol && return x
                if have_x2 && abs(x-x2) <= max(abs(x),abs(x2)) * cbrt(eps(float(T)))^2
                    @warn "invlogccdf_newton: 2-cycle detected, falling back to bisection" lp=lp maxlog=1
                    return quantile_bisect(d, -expm1(lp), minmax(x, x0)...)
                end
                x2 = x0
                have_x2 = true
                x0 = x
                x = x0 + exp(lp - logpdf(d,x0) + logexpm1(max(logccdf(d,x0)-lp,0)))
            end
        else
            x = x0 - exp(lp - logpdf(d,x0) + log1mexp(min(logccdf(d,x0)-lp,0)))
            for _ in 1:max_iter
                abs(x-x0) <= max(abs(x),abs(x0)) * tol && return x
                if have_x2 && abs(x-x2) <= max(abs(x),abs(x2)) * cbrt(eps(float(T)))^2
                    @warn "invlogccdf_newton: 2-cycle detected, falling back to bisection" lp=lp maxlog=1
                    return quantile_bisect(d, -expm1(lp), minmax(x, x0)...)
                end
                x2 = x0
                have_x2 = true
                x0 = x
                x = x0 - exp(lp - logpdf(d,x0) + log1mexp(min(logccdf(d,x0)-lp,0)))
            end
        end
        @warn "invlogccdf_newton: maximum iterations reached, falling back to bisection" lp=lp maxlog=1
        return quantile_bisect(d, -expm1(lp), minmax(x, x0)...)
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
