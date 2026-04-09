# Various algorithms for computing quantile

"""
quantile_bisect(d::ContinuousUnivariateDistribution, p::Real, lx::T, rx::T) where {T<:Real}

Compute the quantile of `d` at probability `p` using the bisection method, starting from
the bracketing interval `[lx, rx]`.

In theory, the interval `[lx, rx]` should satisfy `cdf(d, lx) <= p <= cdf(d, rx)`.
However, due to numerical issues, this condition may not hold. In such cases, the algorithm
attempts to recover a valid bracketing interval by expanding the initial interval.
"""
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

    # find a valid bracketing interval, if necessary
    lx, rx = find_interval_quantile_bisect(d, p, lx, rx)

    # find quantile using bisect algorithm
    while rx - lx > tol
        m = (lx + rx)/2
        c = cdf(d, m)
        if p > c
            lx = m
        else
            rx = m
        end
    end
    return (lx + rx)/2
end

@inline function find_interval_quantile_bisect(d::ContinuousUnivariateDistribution, p::Real, x_left::T, x_right::T) where {T<:Real}
    c_left = cdf(d, x_left)
    c_right = cdf(d, x_right)
    @show c_left c_right
    c_left <= p <= c_right && return x_left, x_right
    # expand the interval by eps() * 2^i at each iteration until it contains the quantile.
    max_expand = 64 # max `i` in `eps() * 2^i`. Completely arbitrary to avoid infinite loop.
    if p > c_right
        step = max(abs(x_right), one(x_right)) * eps(float(T))
        for i in 1:max_expand
            x_right += step
            c_right = cdf(d, x_right)
            if p <= c_right
                break
            end
            step *= 2
        end
    end
    if p < c_left
        step = max(abs(x_left), one(x_left)) * eps(float(T))
        for i in 1:max_expand
            x_left -= step
            c_left = cdf(d, x_left)
            if p >= c_left
                break
            end
            step *= 2
        end
    end

    # infinite loop in bisect if isinf(x_right). Still fine is isinf(x_left) though
    if (!isinf(x_right)) && (c_left <= p <= c_right)
        return x_left, x_right
    end
    throw(ArgumentError("[$x_left, $x_right] is not a valid bracketing interval for `quantile(d, $p)`"))
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

function quantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12)
    x = xs + (p - cdf(d, xs)) / pdf(d, xs)
    T = typeof(x)
    if 0 < p < 1
        x0 = T(xs)
        while abs(x-x0) > max(abs(x),abs(x0)) * tol
            x0 = x
            x = x0 + (p - cdf(d, x0)) / pdf(d, x0)
        end
        return x
    elseif p == 0
        return T(minimum(d))
    elseif p == 1
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function cquantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12)
    x = xs + (ccdf(d, xs)-p) / pdf(d, xs)
    T = typeof(x)
    if 0 < p < 1
        x0 = T(xs)
        while abs(x-x0) > max(abs(x),abs(x0)) * tol
            x0 = x
            x = x0 + (ccdf(d, x0)-p) / pdf(d, x0)
        end
        return x
    elseif p == 1
        return T(minimum(d))
    elseif p == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogcdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d), tol::Real=1e-12)
    T = typeof(lp - logpdf(d,xs))
    if -Inf < lp < 0
        x0 = T(xs)
        if lp < logcdf(d,x0)
            x = x0 - exp(lp - logpdf(d,x0) + logexpm1(max(logcdf(d,x0)-lp,0)))
            while abs(x-x0) >= max(abs(x),abs(x0)) * tol
                x0 = x
                x = x0 - exp(lp - logpdf(d,x0) + logexpm1(max(logcdf(d,x0)-lp,0)))
            end
        else
            x = x0 + exp(lp - logpdf(d,x0) + log1mexp(min(logcdf(d,x0)-lp,0)))
            while abs(x-x0) >= max(abs(x),abs(x0))*tol
                x0 = x
                x = x0 + exp(lp - logpdf(d,x0) + log1mexp(min(logcdf(d,x0)-lp,0)))
            end
        end
        return x
    elseif lp == -Inf
        return T(minimum(d))
    elseif lp == 0
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function invlogccdf_newton(d::ContinuousUnivariateDistribution, lp::Real, xs::Real=mode(d), tol::Real=1e-12)
    T = typeof(lp - logpdf(d,xs))
    if -Inf < lp < 0
        x0 = T(xs)
        if lp < logccdf(d,x0)
            x = x0 + exp(lp - logpdf(d,x0) + logexpm1(max(logccdf(d,x0)-lp,0)))
            while abs(x-x0) >= max(abs(x),abs(x0)) * tol
                x0 = x
                x = x0 + exp(lp - logpdf(d,x0) + logexpm1(max(logccdf(d,x0)-lp,0)))
            end
        else
            x = x0 - exp(lp - logpdf(d,x0) + log1mexp(min(logccdf(d,x0)-lp,0)))
            while abs(x-x0) >= max(abs(x),abs(x0)) * tol
                x0 = x
                x = x0 - exp(lp - logpdf(d,x0) + log1mexp(min(logccdf(d,x0)-lp,0)))
            end
        end
        return x
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
