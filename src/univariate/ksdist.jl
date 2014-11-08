# This program is a Julia port of KolmogorovSmirnovDist.c, which is 
# distributed under the GNU GPL v3 license.
# The original code is available from http://simul.iro.umontreal.ca/ksdir/
#
# /********************************************************************
#  *
#  * File:          KolmogorovSmirnovDist.c
#  * Environment:   ISO C99 or ANSI C89
#  * Author:        Richard Simard
#  * Organization:  DIRO, Université de Montréal
#  * Date:          1 February 2012
#  * Version        1.1
# 
#  * Copyright 1 march 2010 by Université de Montréal,
#                              Richard Simard and Pierre L'Ecuyer
#  =====================================================================
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, version 3 of the License.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
#  =====================================================================*/


# Distribution of the (two-sided) Kolmogorov-Smirnoff statistic
#   D_n = \sup_x |\hat{F}_n(x) - F(x)|
#   sqrt(n) D_n converges a.s. to the Kolmogorov distribution.
immutable KSDist <: ContinuousUnivariateDistribution
    n::Int
end

# support handling

isupperbounded(::Union(KSDist, Type{KSDist})) = true
islowerbounded(::Union(KSDist, Type{KSDist})) = true
isbounded(::Union(KSDist, Type{KSDist})) = true

minimum(d::KSDist) = 1 / (2 * d.n)
maximum(d::KSDist) = 1.0
insupport(d::KSDist, x::Real) = minimum(d) <= x <= 1.0

const NEXACT = 500
const NKOLMO = 100_000

# TODO: implement Simard and L'Ecuyer (2011) meta-algorithm
# requires Pomeranz and Pelz-Good algorithms
function cdf(d::KSDist,x::Real)
    n = d.n
    b = x*n
    # known exact values
    if b <= 0.5
        return 0.0
    elseif b <= 1.0
        # accuracy could be improved
        return exp(lfact(n)+n*(log(2.0*b-1.0)-log(n)))
    elseif x >= 1.0
        return 1.0
    elseif b >= n-1
        return 1.0 - 2.0*(1.0-x)^n
    end

    a = b*x
    if a >= 18.0
        return 1.0
    elseif n <= 10_000
        if a <= 4.0
            return cdf_durbin(d,x)
        else
            return 1.0 - ccdf_miller(d,x)
        end
    else
        return cdf(Kolmogorov(),sqrt(a))
    end
end

# Simard and L'Ecuyer (2011) meta-algorithm uses several algorithms
# to compute exact or approximated values.  Dividing the argument
# space into subspaces, a suitable algorithm is selected from the
# viewpoint of precision and computing cost.
#
#   Simard, Richard, and Pierre L’Ecuyer.
#   "Computing the two-sided Kolmogorov-Smirnov distribution."
#   Journal of Statistical Software 39.11 (2011): 1-18.
#
#   http://www.jstatsoft.org/v39/i11/paper
#
#            Section                       Method
#   ----------------------------------------------------------
#   2. Exact methods
#     2.1. Ruben and Gambino (1982)    cdf_ruben_gambino(d, x)
#     2.2. Pomeranz (1974)             cdf_pomeranz(d, x)
#     2.3. Durbin (1973)               cdf_durbin(d, x)
#   3. Asymptotic approximations
#          Pelz and Good (1976)        cdf_pelz_good(d, x)
#
function cdf2(d::KSDist, x::Real)
    # debug
    n = d.n
    u = cdf_ruben_gambino(d, x)

    if !isnan(u)
        return u
    end

    w = n * x * x
    if n <= NEXACT
        if w < 0.754693
            return cdf_durbin(d, x)
        elseif w < 4.0
            return cdf_pomeranz(d, x)
        end
        return 1.0 - ccdf2(d, x)
    elseif w * x * n <= 7.0 && n <= NKOLMO
        return cdf_durbin(d, x)
    end

    return cdf_pelz_good(d, x)
end

function ccdf(d::KSDist,x::Real)    
    n = d.n
    b = x*n
    # Ruben and Gambino (1982) known exact values
    if b <= 0.5
        return 1.0
    elseif b <= 1.0
        return 1.0-exp(lfact(n)+n*(log(2.0*b-1.0)-log(n)))
    elseif x >= 1.0
        return 0.0
    elseif b >= n-1
        return 2.0*(1.0-x)^n
    end

    a = b*x
    if a >= 370.0
        return 0.0
    elseif a >= 4.0 || (n > 140 && a >= 2.2)
        return ccdf_miller(d,x)
    else
        return 1.0-cdf(d,x)
    end
end

function ccdf2(d::KSDist, x::Real)
    n = d.n
    u = ccdf_ruben_gambino(d, x)

    if !isnan(u)
        return u
    end

    w = n * x * x
    if n <= NEXACT
        if w < 4.0
            return 1.0 - cdf(d, x)
        else
            return ccdf_miller(d, x)
        end
    elseif w >= 2.65
        return ccdf_miller(d, x)
    end

    return 1.0 - cdf2(d, x)
end

# Durbin matrix CDF method, based on Marsaglia, Tsang and Wang (2003)
# modified to avoid need for exponent tracking
function cdf_durbin(d::KSDist,x::Float64)
    #debug
    warn("cdf_durbin")
    n = d.n
    k, ch, h = ceil_rems_mult(n,x)

    m = 2*k-1
    H = Array(Float64,m,m)    
    for i = 1:m, j = 1:m
        H[i,j] = i-j+1 >= 0 ? 1.0 : 0.0        
    end
    r = 1.0
    for i = 1:m 
        # (1-h^i) = (1-h)(1+h+...+h^(i-1))
        H[i,1] = H[m,m-i+1] = ch*r
        r += h^i
    end
    H[m,1] += h <= 0.5 ? -h^m : -h^m+(h-ch)
    for i = 1:m, j = 1:m
        for g = 1:max(i-j+1,0)
            H[i,j] /= g
        end
        # we can avoid keeping track of the exponent by dividing by e
        # (from Stirling's approximation)
        H[i,j] /= e
    end
    Q = H^n
    s = Q[k,k]
    s*stirling(n)
end

# Ruben-Gambino's exact formulae
function cdf_ruben_gambino(d::KSDist, x::Float64)
    #debug
    warn("cdf_ruben_gambino")
    n = d.n

    if n * x * x >= 18.0 || x >= 1.0
        return 1.0
    elseif x <= 0.5 / n
        return 0.0
    elseif n == 1
        return 2x - 1.0
    elseif x <= 1.0 / n
        # n!(2x - 1/n)^n
        t = 2.0 * x * n - 1.0
        if n <= NEXACT
            w = rapfac(n)
            return w * t^n
        else
            w = lfact(n) + n * log(t / n)
            return exp(w)
        end
    elseif x >= 1.0 - 1.0 / n
        return 1.0 - 2(1.0 - x)^n
    end

    # cannot compute exact value, give up
    return NaN
end

function ccdf_ruben_gambino(d::KSDist, x::Float64)
    #debug
    warn("ccdf_ruben_gambino")
    n = d.n
    w = n * x * x
    if w >= 370.0 || x >= 1.0
        return 0.0
    elseif w <= 0.0274 || x <= 0.5 / n
        return 1.0
    elseif n == 1
        return 2.0 - 2.0 * x
    end

    if x <= 1.0 / n
        t = 2.0 * x * n - 1.0
        if n <= NEXACT
            z = rapfac(n)
            warn("one")
            return 1.0 - z * t^n
        end
        z = lfact(n) + n * log(t / n)
        warn("two")
        return 1.0 - exp(z)
    elseif x >= 1.0 - 1.0 / n
        warn("three")
        return 2.0 * (1.0 - x)^n
    end

    # cannot compute exact value, give up
    warn("four")
    return NaN
end

# Pelz-Good's asymptotic approximation
function cdf_pelz_good(d::KSDist, x::Float64)
    #debug
    warn("cdf_pelz_good")
    # constants
    n = d.n
    JMAX = 20
    EPS = 1.0e-10
    C = 2.506628274631001  # sqrt(2*pi)
    C2 = 1.2533141373155001  # sqrt(pi/2)
    PI = pi
    PI2 = PI * PI
    PI4 = PI2 * PI2
    RACN = sqrt(n)
    z = RACN * x
    z2 = z * z
    z4 = z2 * z2
    z6 = z4 * z2
    w = PI2 / (2.0 * z * z)

    term = 1.0
    j = 0
    sum = 0.0
    local ti
    while j <= JMAX && term > EPS * sum
        ti = j + 0.5
        term = exp(-ti * ti * w)
        sum += term
        j += 1
    end
    sum *= C / z

    term = 1.0
    tom = 0.0
    j = 0
    while j <= JMAX && abs(term) > EPS * abs(tom)
        ti = j + 0.5
        term = (PI2 * ti * ti - z2) * exp(-ti * ti * w)
        tom += term
        j += 1
    end
    sum += tom * C2 / (RACN * 3.0 * z4)

    term = 1.0
    tom = 0.0
    j = 0
    while j <= JMAX && abs(term) > EPS * abs(tom)
        ti = j + 0.5
        term = (6z6 + 2z4 + PI2 * (2z4 - 5z2) * ti * ti +
            PI4 * (1.0 - 2z2) * ti * ti * ti * ti)
        term *= exp(-ti * ti * w)
        tom += term
        j += 1
    end
    sum += tom * C2 / (n * 36.0 * z * z6)

    term = 1.0
    tom = 0.0
    j = 1
    while j <= JMAX && term > EPS * tom
        ti = j
        term = PI2 * ti * ti * exp(-ti * ti * w)
        tom += term
        j += 1
    end
    sum -= tom * C2 / (n * 18.0 * z * z2)

    term = 1.0
    tom = 0.0
    j = 0
    while j <= JMAX && abs(term) > EPS * abs(tom)
        ti = j + 0.5
        ti = ti * ti
        term = (-30z6 - 90z6 * z2 + PI * (135z4 - 96z6) * ti +
            PI4 * (212z4 - 60z2) * ti * ti + PI2 * PI4 * ti * ti * ti * (5.0 - 30z2))
        term *= exp(-ti * w)
        tom += term
        j += 1
    end
    sum += tom * C2 / (RACN * n * 3240.0 * z4 * z6)

    term = 1.0
    tom = 0.0
    j = 1
    while j <= JMAX && abs(term) > EPS * abs(tom)
        ti = j * j
        term = (3.0 * PI2 * ti * z2 - PI4 * ti * ti) * exp(-ti * w)
        tom += term
        j += 1
    end
    sum += tom * C2 / (RACN * n * 108.0 * z6)

    return sum
end

# The Pomeranz recursion formula (1974)
function cdf_pomeranz(d::KSDist, x::Float64)
    #debug
    warn("cdf_pomeranz")
    # constants
    EPS = 1.0e-15
    ENO = 350
    RENO = ldexp(1.0, ENO)
    n = d.n
    t = n * x

    coreno = 1
    A = zeros(2 * n + 2)
    Atflo = zeros(2 * n + 2)
    Atcei = zeros(2 * n + 2)
    V = zeros(Float64, (n + 2, 2))
    H = Array(Float64, (n + 2, 4))

    V[1,2] = RENO

    # decompose t = l + f s.t.
    #   l: non-negative integer
    #   f: 0 ≤ f < 1
    l = ifloor(t)
    f = t - l
    stop = 2n + 2

    # calculate floors and ceils
    if f > 0.5
        # Case (iii): f = 0
        for i in 1:2:stop
            Atflo[i] = div(i, 2) - 1 - l
            Atcei[i] = div(i, 2) + 1 + l
        end

        for i in 2:2:stop
            Atflo[i] = div(i, 2) - 2 - l
            Atcei[i] = div(i, 2) + l
        end
    elseif f > 0.0
        # Case (ii): 0 < f ≤ 1/2
        for i in 1:stop
            Atflo[i] = div(i, 2) - 1 - l
        end

        Atcei[1] = 1 + l
        for i in 2:stop
            Atcei[i] = div(i, 2) + l
        end
    else
        # Case (i): f = 0
        for i in 1:2:stop
            Atflo[i] = div(i, 2) - l
            Atcei[i] = div(i, 2) + l
        end

        for i in 2:2:stop
            Atflo[i] = div(i, 2) - 1 - l
            Atcei[i] = div(i, 2) - 1 + l
        end
    end

    # calculate A
    A[1] = 0.0
    A[2] = min(t - floor(t), ceil(t) - t)
    A[3] = 1.0 - A[2]
    for i in 4:stop-1
        A[i] = A[i-2] + 1
    end
    A[stop] = n

    H[1,1] = 1.0
    w = 2.0 * A[2] / n
    for j in 2:n+2
        H[j,1] = w * H[j-1,1] / (j - 1)
    end

    H[1,2] = 1.0
    w = (1.0 - 2.0 * A[2]) / n
    for j in 2:n+2
        H[j,2] = w * H[j-1,2] / (j - 1)
    end

    H[1,3] = 1.0
    w = A[2] / n
    for j in 2:n+2
        H[j,3] = w * H[j-1,3] / (j - 1)
    end

    H[1,4] = 1.0
    for j in 2:n+2
        H[j,4] = 0.0
    end

    # indices i and i-1 for V[i][]
    r1 = 1
    r2 = 2

    for i in 2:2n+2
        # j = floor{Ai - t} + 2, …, ceil{Ai + t}
        jlo = max(1, Atflo[i] + 2)
        jhi = min(n + 1, Atcei[i])

        # k = max(1, floor{Ai-1 - t} + 2), …, min(j, ceil{Ai-1 + t})
        klo = max(1, Atflo[i-1] + 2)

        w = (A[i] - A[i-1]) / n
        s = 0
        for j in 1:4
            if abs(w - H[2,j]) <= EPS
                s = j
                break
            end
        end
        @assert s > 0

        minsum = RENO
        r1, r2 = r2, r1  # r1: i-1, r2: i

        for j in jlo:jhi
            khi = min(j, Atcei[i-1])
            sum = 0.0
            for k in khi:-1:klo
                sum += V[k,r1] * H[j-k+1,s]
            end
            V[j,r2] = sum
            minsum = min(minsum, sum)
        end

        if minsum < 1.0e-280
            for j in jlo:jhi
                V[j,r2] *= RENO
            end
            coreno += 1
        end
    end

    sum = V[n+1,r2]
    w = lfact(n) - coreno * ENO * log(2) + log(sum)
    return w >= 0.0 ? 1.0 : exp(w)
end

# Miller (1956) approximation
function ccdf_miller(d::KSDist, x::Real)
    2.0*ccdf(KSOneSided(d.n),x)
end

## these functions are used in durbin and pomeranz algorithms
# calculate exact remainders the easy way
function floor_rems_mult(n,x)
    t = big(x)*big(n)
    fl = floor(t)
    lrem = t - fl
    urem = (fl+one(fl)) - t
    return convert(typeof(n),fl), convert(typeof(x),lrem), convert(typeof(x),urem)
end
function ceil_rems_mult(n,x)
    t = big(x)*big(n)
    cl = ceil(t)
    lrem = t - (cl - one(cl))
    urem = cl - t
    return convert(typeof(n),cl), convert(typeof(x),lrem), convert(typeof(x),urem)
end

# n!*(e/n)^n
function stirling(n)    
    if n < 500
        s = 1.0
        for i = 1:n
            s *= i/n*e 
        end
        return s
    else
        # 3rd-order Stirling's approximation more accurate for large n
        twn = 12.0*n
        return sqrt(2.0*pi*n)*(1.0 + twn\(1 + (2.0*twn)\(1 - (15.0*twn)\139.0)))
    end
end

# n! / n^n
function rapfac(n)
    res = 1.0 / n
    for i in 2:n
        res *= i / n
    end
    return res
end
