"""
    NoncentralChisq(ν, λ)

The *noncentral chi-squared distribution* with `ν` degrees of freedom and noncentrality parameter `λ` has the probability density function

```math
f(x; \\nu, \\lambda) = \\frac{1}{2} e^{-(x + \\lambda)/2} \\left( \\frac{x}{\\lambda} \\right)^{\\nu/4-1/2} I_{\\nu/2-1}(\\sqrt{\\lambda x}), \\quad x > 0
```

It is the distribution of the sum of squares of `ν` independent [`Normal`](@ref) variates with individual means ``\\mu_i`` and

```math
\\lambda = \\sum_{i=1}^\\nu \\mu_i^2
```

```julia
NoncentralChisq(ν, λ)     # Noncentral chi-squared distribution with ν degrees of freedom and noncentrality parameter λ

params(d)    # Get the parameters, i.e. (ν, λ)
```

External links

* [Noncentral chi-squared distribution on Wikipedia](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution)
"""
struct NoncentralChisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    λ::T
    NoncentralChisq{T}(ν::T, λ::T) where {T<:Real} = new{T}(ν, λ)
end

function NoncentralChisq(ν::T, λ::T; check_args::Bool=true) where {T<:Real}
    @check_args NoncentralChisq (ν, ν > zero(ν)) (λ, λ >= zero(λ))
    return NoncentralChisq{T}(ν, λ)
end

NoncentralChisq(ν::Real, λ::Real; check_args::Bool=true) = NoncentralChisq(promote(ν, λ)...; check_args=check_args)
NoncentralChisq(ν::Integer, λ::Integer; check_args::Bool=true) = NoncentralChisq(float(ν), float(λ); check_args=check_args)

@distr_support NoncentralChisq 0.0 Inf

#### Conversions

function convert(::Type{NoncentralChisq{T}}, ν::S, λ::S) where {T<:Real,S<:Real}
    NoncentralChisq(T(ν), T(λ))
end
function Base.convert(::Type{NoncentralChisq{T}}, d::NoncentralChisq) where {T<:Real}
    NoncentralChisq{T}(T(d.ν), T(d.λ))
end
Base.convert(::Type{NoncentralChisq{T}}, d::NoncentralChisq{T}) where {T<:Real} = d

### Parameters

params(d::NoncentralChisq) = (d.ν, d.λ)
@inline partype(d::NoncentralChisq{T}) where {T<:Real} = T


### Statistics

mean(d::NoncentralChisq) = d.ν + d.λ
var(d::NoncentralChisq) = 2(d.ν + 2d.λ)
skewness(d::NoncentralChisq) = 2sqrt2 * (d.ν + 3d.λ) / sqrt(d.ν + 2d.λ)^3
kurtosis(d::NoncentralChisq) = 12(d.ν + 4d.λ) / (d.ν + 2d.λ)^2

function mgf(d::NoncentralChisq, t::Real)
    exp(d.λ * t / (1 - 2t)) * (1 - 2t)^(-d.ν / 2)
end

function cf(d::NoncentralChisq, t::Real)
    cis(d.λ * t / (1 - 2im * t)) * (1 - 2im * t)^(-d.ν / 2)
end


### Evaluation & Sampling

@_delegate_statsfuns NoncentralChisq nchisq ν λ
# Code for RFunctions dependency
# @rand_rdist(NoncentralChisq)

# According to Wikipedia, if ν=1, then the support should be (0,Inf) rather than [0,Inf)
@distr_support NoncentralChisq 0.0 Inf

# The code below is ported from the R code written in C.
# The variable names are as written in the R code, and it should be easy to
# compare the code in R with the code in C.

function cdf(d::NoncentralChisq, x::Real)
    return pnchisq(x, d.ν, d.λ, true, false)
end

function logcdf(d::NoncentralChisq, x::Real)
    return pnchisq(x, d.ν, d.λ, true, true)
end

function pnchisq(x::Real, df::Real, ncp::Real, lower_tail::Integer, log_p::Integer)
    DBL_EPSILON = eps(Float64)
    R_D__0 = log_p ? -Inf : 0.0
    R_D__1 = log_p ? 0.0 : 1.0

    if isnan(x) || isnan(df) || isnan(ncp)
        return x + df + ncp
    end
    if isinf(df) || isinf(ncp)
        throw(ArgumentError("ν and λ must be finite"))
    end
    if df < 0.0 || ncp < 0.0
        throw(ArgumentError("ν and λ must be nonnegative"))
    end
    ans = pnchisq_raw(x, df, ncp, 1e-12, 8 * DBL_EPSILON, 1000000, lower_tail, log_p)

    if x <= 0.0 || isinf(x)
        return ans
    end

    if ncp >= 80
        if lower_tail
            ans = min(ans, R_D__1)
        else
            if ans < (log_p ? -10.0 * log(10) : 1e-10)
                @warn "Precision warning."
            end
            if !log_p && ans < 0.0
                ans = 0.0
            end
        end
    end

    if !log_p || ans < -1e-8
        return ans
    else
        ans = pnchisq_raw(x, df, ncp, 1e-12, 8 * DBL_EPSILON, 1000000, !lower_trail, false)
        return log1p(-ans)
    end
end


# For logspace_add defined below
# using LogExpFunctions

function pnchisq_raw(x::Real, f::Real, theta::Real, errmax::Real, reltol::Real, itrmax::Integer, lower_tail::Integer, log_p::Integer)
    DBL_EPSILON = eps(Float64)
    _dbl_min_exp = -708.3964
    _dbl_min_exp = -1021.0
    R_D__0 = log_p ? -Inf : 0.0
    R_D__1 = log_p ? 0.0 : 1.0
    function R_D_exp(_L)
        return log_p ? x : exp(x)
    end
    function R_Log1_Exp(x)
        return x > -log(2) ? log(-expm1(x)) : log1p(-exp(x))
    end

    R_D_val(x) = log_p ? log(x) : x
    R_D_Clog(p) = log_p ? log1p(-p) : p
    R_DT_val(x) = lower_tail ? R_D_val(x) : R_D_Clog(x)

    R_DT_0 = lower_tail ? R_D__0 : R_D__1
    R_DT_1 = lower_tail ? R_D__1 : R_D__0
    M_LN2 = log(2)
    M_LN_SQRT_2PI = log(sqrt(2 * pi))
    function logspace_add(logx, logy)
        return log(exp(logx) + exp(logy))
        # return logaddexp(logx, logy)
    end

    # The R code uses both lgamma, a function defined in ISOC99, and lgammafn, a function defined in nmath/lgamma.c
    # The documentation of both say they compute the log of the absolute value of gamma(x), so I am not sure why both functions are called
    # Documentation for lgamma
    # https://en.cppreference.com/w/c/numeric/math/lgamma 
    function lgamma(x::Real)
        return logabsgamma(x)[1]
    end
    # Documentation for lgammafn
    # https://github.com/SurajGupta/r-source/blob/master/src/nmath/lgamma.c
    function lgammafn(x::Real)
        return logabsgamma(x)[1]
    end

    l_lam = -1.0
    l_x = -1.0
    lu = -1

    if x <= 0.0
        if x == 0.0 && f == 0.0
            _L = -0.5 * theta
            return lower_tail ? R_D_exp(_L) : (log_p ? R_Log1_Exp(_L) : -expm1(_L))
        end
        return R_DT_0
    end
    if isinf(x)
        return R_DT_1
    end

    if theta < 80
        if lower_tail && f > 0.0 && log(x) < M_LN2 + 2 / f * (lgamma(f / 2.0 + 1) + _dbl_min_exp)
            lambda = 0.5 * theta
            pr = -lambda
            log_lam = log(lambda)
            sum = sum2 = -Inf
            i = 0
            while i < 110
                sum2 = logspace_add(sum2, pr)
                temp = lower_tail ? cdf(Chisq(f + 2 * i), x) : 1 - cdf(Chisq(f + 2 * i), x)
                sum = logspace_add(sum, pr + log(temp))
                if sum >= -1e-15
                    break
                end
                i += 1
                pr += log_lam - log(i)
            end
            ans = sum - sum2
            return log_p ? ans : exp(ans)
        else
            lambda = 0.5 * theta
            sum = 0
            sum2 = 0
            pr = exp(-lambda)

            i = 0
            while (i < 110)
                sum2 += pr
                temp = lower_tail ? cdf(Chisq(f + 2 * i), x) : 1 - cdf(Chisq(f + 2 * i), x)
                sum += pr * temp
                if sum2 >= 1 - 1e-15
                    break
                end
                i += 1
                pr *= lambda / i
            end
            ans = sum / sum2
            return log_p ? log(ans) : ans
        end
    end

    lam = 0.5 * theta
    lamSml = (-lam < _dbl_min_exp)
    if lamSml
        u = 0
        lu = -lam
        l_lam = log(lam)
    else
        u = exp(-lam)
    end

    v = u
    x2 = 0.5 * x
    f2 = 0.5 * f
    f_x_2n = f - x

    # t gets assigned below in all cases
    t = x2 - f2
    if f2 * DBL_EPSILON > 0.125 && abs(t) < sqrt(DBL_EPSILON) * f2
        lt = (1 - t) * (2 - t / (f2 + 1)) - M_LN_SQRT_2PI - 0.5 * log(f2 + 1)
    else
        lt = f2 * log(x2) - x2 - lgammafn(f2 + 1)
    end
    tSml = (lt < _dbl_min_exp)
    if tSml
        if x > f + theta + 5 * sqrt(2 * (f + 2 * theta))
            return R_DT_1
        end
        l_x = log(x)
        ans = term = 0.0
        t = 0
    else
        t = exp(lt)
        ans = term = v * t
    end

    n = 1
    f_2n = f + 2.0
    f_x_2n += 2.0
    while n <= itrmax
        if f_x_2n > 0
            bound = t * x / f_x_2n
            is_r = false
            is_b = (bound <= errmax)
            is_r = (term <= reltol * ans)
            if is_b && is_r
                break
            end
        end

        if lamSml
            lu += l_lam - log(n)
            if lu >= _dbl_min_exp
                v = u = exp(lu)
                lamSml = false
            end
        else
            u *= lam / n
            v += u
        end
        if tSml
            lt += l_x - log(f_2n)
            if lt >= _dbl_min_exp
                t = exp(lt)
                tSml = false
            end
        else
            t *= x / f_2n
        end
        if !lamSml && !tSml
            term = v * t
            ans += term
        end

        n += 1
        f_2n += 2
        f_x_2n += 2
    end

    if n > itrmax
        throw(ErrorException("No convergence after $itrmax (itrmax) iterations."))
    end
    dans = ans
    return R_DT_val(dans)
end

function pdf(d::NoncentralChisq, x::Real)
    return dnchisq(x, d.ν, d.λ, false)
end

function logpdf(d::NoncentralChisq, x::Real)
    return dnchisq(x, d.ν, d.λ, true)
end

function dnchisq(x::Real, df::Real, ncp::Real, give_log::Integer)
    R_D__0 = give_log ? -Inf : 0.0

    eps = 5e-15

    if isnan(x) || isnan(df) || isnan(ncp)
        return x + df + ncp
    end
    if isinf(df) || isinf(ncp) || ncp < 0 || df < 0
        throw(ArgumentError("ν and λ must be finite and nonnegative"))
    end
    if x < 0
        return R_D__0
    end
    if x == 0 && df < 2.0
        return Inf
    end
    if ncp == 0
        return df > 0 ? (give_log ? logpdf(Chisq(df), x) : pdf(Chisq(df), x)) : R_D__0
    end
    if x == Inf
        return R_D__0
    end

    ncp2 = 0.5 * ncp

    imax = ceil((-(2 + df) + sqrt((2 - df) * (2 - df) + 4 * ncp * x)) / 4)
    if (imax < 0)
        imax = 0.0
    end
    if isfinite(imax)
        dfmid = df + 2 * imax
        mid = pdf(Poisson(ncp2), imax) * pdf(Chisq(dfmid), x)
    else
        mid = 0.0
    end

    if mid == 0.0
        if (give_log || ncp > 1000.0)
            n1 = df + ncp
            ic = n1 / (n1 + ncp)
            return give_log ? logpdf(Chisq(n1 * ic), x * ic) : pdf(Chisq(n1 * ic), x * ic)
        else
            return R_D__0
        end
    end

    sum = mid

    term = mid
    df = dfmid
    i = imax
    x2 = x * ncp2

    while true
        i += 1
        q = x2 / i / df
        df += 2.0
        term *= q
        sum += term
        (q >= 1.0 || term * q > (1.0 - q) * eps || term > 1e-10 * sum) || break
    end

    term = mid
    df = dfmid
    i = imax
    while i != 0
        df -= 2.0
        q = i * df / x2
        i -= 1
        term *= q
        sum += term
        if q < 1.0 && term * q <= (1.0 - q) * eps
            break
        end
    end

    R_D_val = give_log ? log(sum) : sum
    return R_D_val
end

function quantile(d::NoncentralChisq, x::Real)
    return qnchisq(x, d.ν, d.λ, true, false)
end

function qnchisq(p::Real, df::Real, ncp::Real, lower_tail::Integer, log_p::Integer)
    DBL_MAX = prevfloat(typemax(Float64))
    DBL_MIN = nextfloat(typemin(Float64))
    DBL_EPSILON = eps(Float64)
    function R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)
        if (log_p)
            if p > 0
                throw(ArgumentError("NAN"))
            end
            if p == 0
                return lower_tail ? _RIGHT_ : _LEFT_
            end
            if p == -Inf
                return lower_tail ? _LEFT_ : _RIGHT_
            end
        else
            if p < 0 || p > 1
                throw(ArgumentError("NAN"))
            end
            if p == 0
                return lower_tail ? _LEFT_ : _RIGHT_
            end
            if p == 1
                return lower_tail ? _RIGHT_ : _LEFT_
            end
        end
    end
    function R_D_qIv(p)
        return log_p ? exp(p) : p
    end

    accu = 1e-13
    racc = 4 * DBL_EPSILON
    Eps = 1e-11
    rEps = 1e-10

    if isnan(p) || isinf(df) || isnan(ncp)
        return p + df + ncp
    end
    if isinf(df)
        throw(ArgumentError("NAN"))
    end
    if df < 0.0 || ncp < 0.0
        throw(ArgumentError("NAN"))
    end

    R_Q_P01_boundaries(p, 0, Inf)

    pp = R_D_qIv(p)
    if pp > 1 - DBL_EPSILON
        return lower_tail ? Inf : 0.0
    end

    b = (ncp * ncp) / (df + 3 * ncp)
    c = (df + 3 * ncp) / (df + 2 * ncp)
    ff = (df + 2 * ncp) / (c * c)
    # TODO fix if we ever need lower_tail to be false or log_p to be true
    temp = quantile(Chisq(ff), p)
    ux = b + c * temp
    if ux <= 0.0
        ux = 1
    end
    ux0 = ux

    if !lower_tail && ncp >= 80
        if pp < 1e-10
            @warn "Precision warning."
        end
        p = log_p ? -expm1(p) : p
        lower_tail = true
    else
        p = pp
    end

    pp = min(1 - DBL_EPSILON, p * (1 + Eps))
    if lower_tail
        while ux < DBL_EPSILON && pnchisq_raw(ux, df, ncp, Eps, rEps, 10000, true, false) < pp
            ux *= 2
        end
        pp = p * (1 - Eps)
        lx = min(ux0, DBL_MAX)
        while lx > DBL_MIN && pnchisq_raw(lx, df, ncp, Eps, rEps, 10000, true, false) > pp
            lx *= 0.5
        end
    else
        while ux < DBL_MAX && pnchisq_raw(ux, df, ncp, Eps, rEps, 10000, false, false) > pp
            ux *= 2
        end
        pp = p * (1 - Eps)
        lx = min(ux0, DBL_MAX)
        while lx > DBL_MIN && pnchisq_raw(lx, df, ncp, Eps, rEps, 10000, false, false) < pp
            lx *= 0.5
        end
    end

    if lower_tail
        while true
            nx = 0.5 * (lx + ux)
            if pnchisq_raw(nx, df, ncp, accu, racc, 100000, true, false) > p
                ux = nx
            else
                lx = nx
            end
            ((ux - lx) / nx > accu) || break
        end
    else
        while true
            nx = 0.5 * (lx + ux)
            if pnchisq_raw(nx, df, ncp, accu, racc, 100000, false, false) < p
                ux = nx
            else
                lx = nx
            end
            ((ux - lx) / nx > accu) || break
        end
    end
    return 0.5 * (ux + lx)
end

function rand(d::NoncentralChisq)
    ν = d.ν
    λ = d.λ
    if isnan(ν) || isinf(λ) || ν < 0.0 || λ < 0.0
        throw(ArgumentError("NAN"))
    end
    if λ == 0.0
        return ν == 0.0 ? 0.0 : rand(Gamma(ν / 2.0, 2.0one(ν)))
    else
        r = rand(Poisson(λ / 2.0))
        if r > 0.0
            r = rand(Chisq(2.0 * r))
        end
        if ν > 0.0
            r += rand(Gamma(ν / 2.0, 2.0one(ν)))
        end
        return r
    end
end

function rand(rng::AbstractRNG, d::NoncentralChisq)
    ν = d.ν
    λ = d.λ
    if isnan(ν) || isinf(λ) || ν < 0.0 || λ < 0.0
        throw(ArgumentError("NAN"))
    end
    if λ == 0.0
        return ν == 0.0 ? 0.0 : rand(rng, Gamma(ν / 2.0, 2.0one(ν)))
    else
        r = rand(rng, Poisson(λ / 2.0))
        if r > 0.0
            r = rand(rng, Chisq(2.0 * r))
        end
        if ν > 0.0
            r += rand(rng, Gamma(ν / 2.0, 2.0one(ν)))
        end
        return r
    end
end

function sampler(d::NoncentralChisq)
    return d
end
