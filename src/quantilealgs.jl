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

mutable struct ValueAndPredecessor{T}
    storage::Tuple{T,T}

    function ValueAndPredecessor{T}(storage::Tuple{T,T}) where {T}
        return new(storage)
    end

    function ValueAndPredecessor{T}() where {T}
        storage = Tuple{T,T}((T(NaN),T(NaN)))
        return new(storage)
    end

    function Base.push!(huh::ValueAndPredecessor{T}, e::T) where {T}
        huh.storage = Tuple{T,T}((huh.storage[2], e))
    end

    function Base.getindex(ValueAndPredecessor::ValueAndPredecessor{T}, delay::Int)::T where {T}
        return ValueAndPredecessor.storage[length(ValueAndPredecessor.storage) + delay]
    end
end

# if starting at mode, Newton is convergent for any unimodal continuous distribution, see:
#   Göknur Giner, Gordon K. Smyth (2014)
#   A Monotonically Convergent Newton Iteration for the Quantiles of any Unimodal
#   Distribution, with Application to the Inverse Gaussian Distribution
#   http://www.statsci.org/smyth/pubs/qinvgaussPreprint.pdf

function newton(f, xs::T=mode(d), tol::Real=1e-12) where {T}
    converged(a,b) = abs(a-b) <= max(abs(a),abs(b)) * tol
    x = ValueAndPredecessor{T}()
    push!(x, xs)
    push!(x, x[0] + f(x[0]))
    while !converged(x[0], x[-1])
        df = f(x[0])
        r = x[0] + df
        # Vanilla Newton algorithm is known to be prone to oscillation,
        # where we, e.g., go from 24.0 to 42.0, back to 24.0 and so forth.
        # We can detect this situation by checking for convergence between
        # the newly-computed "root" and the "root" we had two steps before.
        if converged(r, x[-1])
            # To recover from oscillation, let's use just half of the delta.
            r = r - (df / 2)
        end
        push!(x, r)
    end
    return x[0]
end

function quantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12)
    f(x) = (p - cdf(d, x)) / pdf(d, x)
    # FIXME: can this be expressed via `promote_type()`? Test coverage missing.
    x = xs + f(xs)
    T = typeof(x)
    if 0 < p < 1
        return newton(f, T(xs), tol)
    elseif p == 0
        return T(minimum(d))
    elseif p == 1
        return T(maximum(d))
    else
        return T(NaN)
    end
end

function cquantile_newton(d::ContinuousUnivariateDistribution, p::Real, xs::Real=mode(d), tol::Real=1e-12)
    f(x) = (ccdf(d, x)-p) / pdf(d, x)
    # FIXME: can this be expressed via `promote_type()`? Test coverage missing.
    x = xs + f(xs)
    T = typeof(x)
    if 0 < p < 1
        return newton(f, T(xs), tol)
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
    f_a(x) = -exp(lp - logpdf(d,x) + logexpm1(max(logcdf(d,x)-lp,0)))
    f_b(x) = exp(lp - logpdf(d,x) + log1mexp(min(logcdf(d,x)-lp,0)))
    if -Inf < lp < 0
        x0 = T(xs)
        if lp < logcdf(d,x0)
            return newton(f_a, T(xs), tol)
        else
            return newton(f_b, T(xs), tol)
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
    f_a(x) = exp(lp - logpdf(d,x) + logexpm1(max(logccdf(d,x)-lp,0)))
    f_b(x) = -exp(lp - logpdf(d,x) + log1mexp(min(logccdf(d,x)-lp,0)))
    if -Inf < lp < 0
        x0 = T(xs)
        if lp < logccdf(d,x0)
            return newton(f_a, T(xs), tol)
        else
            return newton(f_b, T(xs), tol)
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
