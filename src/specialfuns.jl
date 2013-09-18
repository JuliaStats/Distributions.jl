# Special functions

realmaxexp{T<:FloatingPoint}(::Type{T}) = with_rounding(()->log(realmax(T)),RoundDown)
realmaxexp(::Type{BigFloat}) = with_bigfloat_rounding(()->log(prevfloat(inf(BigFloat))),RoundDown)

realminexp{T<:FloatingPoint}(::Type{T}) = with_rounding(()->log(realmin(T)),RoundUp)
realmaxexp(::Type{BigFloat}) = with_bigfloat_rounding(()->log(nextfloat(zero(BigFloat))),RoundUp)



# See:
#   Martin Maechler (2012) "Accurately Computing log(1 − exp(− |a|))"
#   http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

# log(1-exp(x)) 
# NOTE: different than Maechler (2012), no negation inside parantheses
log1mexp(x::Real) = x >= -0.6931471805599453 ? log(-expm1(x)) : log1p(-exp(x))
# log(1+exp(x))
log1pexp(x::BigFloat) = x <= realmaxexp(typeof(x)) ? log1p(exp(x)) : x
log1pexp(x::Float64) = x <= 18.0 ? log1p(exp(x)) : x <= 33.3 ? x + exp(-x) : x
log1pexp(x::Float32) = x <= 9f0 ? log1p(exp(x)) : x <= 16f0 ? x + exp(-x) : x
log1pexp(x::Integer) = log1pexp(float(x))
# log(exp(x)-1)
logexpm1(x::BigFloat) = x <= realmaxexp(typeof(x)) ? log(expm1(x)) : x 
logexpm1(x::Float64) = x <= 18.0 ? log(expm1(x)) : x <= 33.3 ? x - exp(-x) : x
logexpm1(x::Float32) = x <= 9f0 ? log(expm1(x)) : x <= 16f0 ? x - exp(-x) : x
logexpm1(x::Integer) = logexpm1(float(x))

φ(z::Real) = exp(-0.5*z*z)/√2π
logφ(z::Real) = -0.5*(z*z + log2π)

Φ(z::Real) = 0.5*erfc(-z/√2)
Φc(z::Real) = 0.5*erfc(z/√2)
logΦ(z::Real) = z < -1.0 ? log(0.5*erfcx(-z/√2)) - 0.5*z*z : log1p(-0.5*erfc(z/√2))
logΦc(z::Real) = z > 1.0 ? log(0.5*erfcx(z/√2)) - 0.5*z*z : log1p(-0.5*erfc(-z/√2))

import Base.Math.@horner

# Rational approximations for the inverse cdf, from:
#   Wichura, M.J. (1988) Algorithm AS 241: The Percentage Points of the Normal Distribution
#   Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 37, No. 3, pp. 477-484
Φinv(p::Integer) = Φinv(float(p))
logΦinv(p::Integer) = logΦinv(float(p))

for (fn,arg) in ((:Φinv,:p),(:logΦinv,:logp))
    @eval begin
        function $fn($arg::Float32)
            if $(fn == :Φinv)
                q = p - 0.5f0
            else
                q = exp(logp) - 0.5f0
            end
            if abs(q) <= 0.425f0 
                r = 0.180625f0 - q*q
                return q * @horner(r,
                                   3.38713_27179f0, 
                                   5.04342_71938f1, 
                                   1.59291_13202f2, 
                                   5.91093_74720f1, 
                                   ) /
                @horner(r,
                        1.0f0,
                        1.78951_69469f1, 
                        7.87577_57664f1, 
                        6.71875_63600f1)  
            else
                if $(fn == :Φinv)
                    if p <= 0f0
                        return p == 0f0 ? -inf(Float32) : nan(Float32)
                    elseif p >= 1f0 
                        return p == 1f0 ? inf(Float32) : nan(Float32)
                    end
                    r = sqrt(q < 0f0 ? -log(p) : -log1p(-p))
                else
                    if logp == -Inf
                        return -inf(Float32)
                    elseif logp >= 0f0 
                        return logp == 0f0 ? inf(Float32) : nan(Float32)
                    end
                    r = sqrt(qf0 < 0 ? -logp : -log1mexp(logp))
                end
                if r < 5.0f0
                    r -= 1.6f0
                    z = @horner(r,
                                1.42343_72777f0, 
                                2.75681_53900f0, 
                                1.30672_84816f0, 
                                1.70238_21103f-1) /
                    @horner(r,
                            1.0f0,
                            7.37001_64250f-1, 
                            1.20211_32975f-1)
                else
                    r -= 5.0f0
                    z = @horner(r,
                                6.65790_51150f0, 
                                3.08122_63860f0, 
                                4.28682_94337f-1, 
                                1.73372_03997f-2) /
                    @horner(r,
                            1.0f0,
                            2.41978_94225f-1, 
                            1.22582_02635f-2) 
                end
                return copysign(z,q)
            end
        end

        function $fn($arg::Float64)
            if $(fn == :Φinv)
                q = p - 0.5
            else
                q = exp(logp) - 0.5
            end
            if abs(q) <= 0.425 
                r = 0.180625 - q*q
                return q * @horner(r,
                                   3.38713_28727_96366_6080e0, 
                                   1.33141_66789_17843_7745e2, 
                                   1.97159_09503_06551_4427e3, 
                                   1.37316_93765_50946_1125e4, 
                                   4.59219_53931_54987_1457e4, 
                                   6.72657_70927_00870_0853e4, 
                                   3.34305_75583_58812_8105e4, 
                                   2.50908_09287_30122_6727e3) /
                @horner(r,
                        1.0,
                        4.23133_30701_60091_1252e1, 
                        6.87187_00749_20579_0830e2, 
                        5.39419_60214_24751_1077e3, 
                        2.12137_94301_58659_5867e4, 
                        3.93078_95800_09271_0610e4, 
                        2.87290_85735_72194_2674e4, 
                        5.22649_52788_52854_5610e3)
            else
                if $(fn == :Φinv)
                    if p <= 0.0
                        return p == 0.0 ? -inf(Float64) : nan(Float64)
                    elseif p >= 1.0 
                        return p == 1.0 ? inf(Float64) : nan(Float64)
                    end
                    r = sqrt(q < 0 ? -log(p) : -log1p(-p))
                else
                    if logp == -Inf
                        return -inf(Float64)
                    elseif logp >= 0.0 
                        return logp == 0.0 ? inf(Float64) : nan(Float64)
                    end
                    r = sqrt(q < 0 ? -logp : -log1mexp(logp))
                end
                if r < 5.0
                    r -= 1.6
                    z = @horner(r,
                                1.42343_71107_49683_57734e0, 
                                4.63033_78461_56545_29590e0, 
                                5.76949_72214_60691_40550e0, 
                                3.64784_83247_63204_60504e0, 
                                1.27045_82524_52368_38258e0, 
                                2.41780_72517_74506_11770e-1, 
                                2.27238_44989_26918_45833e-2, 
                                7.74545_01427_83414_07640e-4) /
                    @horner(r,
                            1.0,
                            2.05319_16266_37758_82187e0, 
                            1.67638_48301_83803_84940e0, 
                            6.89767_33498_51000_04550e-1, 
                            1.48103_97642_74800_74590e-1, 
                            1.51986_66563_61645_71966e-2, 
                            5.47593_80849_95344_94600e-4, 
                            1.05075_00716_44416_84324e-9)
                else
                    r -= 5.0
                    z = @horner(r,
                                6.65790_46435_01103_77720e0, 
                                5.46378_49111_64114_36990e0, 
                                1.78482_65399_17291_33580e0, 
                                2.96560_57182_85048_91230e-1, 
                                2.65321_89526_57612_30930e-2, 
                                1.24266_09473_88078_43860e-3, 
                                2.71155_55687_43487_57815e-5, 
                                2.01033_43992_92288_13265e-7) /
                    @horner(r,
                            1.0,
                            5.99832_20655_58879_37690e-1, 
                            1.36929_88092_27358_05310e-1, 
                            1.48753_61290_85061_48525e-2, 
                            7.86869_13114_56132_59100e-4, 
                            1.84631_83175_10054_68180e-5, 
                            1.42151_17583_16445_88870e-7, 
                            2.04426_31033_89939_78564e-15)            
                end
                return copysign(z,q)
            end
        end
    end
end

# log(x) - x + 1
# fallback
logmxp1(x) = log(x) - x + one(x)
logmxp1(x::Integer) = logmxp1(float(x))

# negative of NSWC DRLOG
function logmxp1(x::Float64)
    if (x < 0.61) || (x > 1.57)
        return log(x) - (x-1.0)
    end
    if x < 0.82
        u = (x-0.7)/0.7
        up2 = u+2.0
        w1 = 0.566749439387323789126387112411845e-01 - u*0.3
    elseif x > 1.18
        t = 0.75*(x-1.0)
        u = t-0.25
        up2 = t+1.75
        w1 = 0.456512608815524058941143273395059e-01 + u/3.0
    else
        u = x-1.0
        up2 = x+1.0
        w1 = 0.0
    end
    r = u/up2
    t = r*r
    z = @horner(t,
                0.7692307692307692307680e-01,
                -0.1505958055914600184836e+00,
                0.9302355725278521726994e-01,
                -0.1787900022182327735804e-01) /
    @horner(t,1.0,
            -0.2824412139355646910683e+01,
            0.2892424216041495392509e+01,
            -0.1263560605948009364422e+01,
            0.1966769435894561313526e+00)
    w = @horner(t,
                0.333333333333333333333333333333333e+00,
                0.200000000000000000000000000000000e+00,
                0.142857142857142857142857142857143e+00,
                0.111111111111111111111111111111111e+00,
                0.909090909090909090909090909090909e-01,
                z)
    return r*(2.0*t*w-u) - w1
end

# negative of NSWC RLOG
function logmxp1(x::Float32)
    if (x < 0.61f0) || (x > 1.57f0)
        return log(x) - (x-1f0)
    end
    if x < 0.82f0
        u = (x-0.7f0)/0.7f0
        up2 = u+2f0
        w1 = 0.566749439387324f-01 - u*0.3f0
    elseif x > 1.18f0
        t = 0.75f0*(x-1f0)
        u = t-0.25f0
        up2 = t+1.75f0
        w1 = 0.456512608815524f-01 + u/3f0
    else
        u = x-1f0
        up2 = x+1f0
        w1 = 0f0
    end
    r = u/up2
    t = r*r
    w = @horner(t,
                0.333333333333333f+00, 
                -.224696413112536f+00,
                0.620886815375787f-02) /
    @horner(t, 1f0,
            -.127408923933623f+01, 
            0.354508718369557f+00)
    return r*(2f0*t*w-u) - w1
end


# negative of NSWC DRLOG1
function log1pmx(x::Float64)
#-----------------------------------------------------------------------
#             EVALUATION OF THE FUNCTION X - LN(1 + X)
#-----------------------------------------------------------------------
#     DOUBLE PRECISION X
#     DOUBLE PRECISION A, B, R, T, U, UP2, W, W1, Z
#     DOUBLE PRECISION P0, P1, P2, P3, Q1, Q2, Q3, Q4
#     DOUBLE PRECISION C1, C2, C3, C4, C5
#-------------------------
#     A = DRLOG (0.7)
#     B = DRLOG (4/3)
#-------------------------
    a = 0.566749439387323789126387112411845e-01
    b = 0.456512608815524058941143273395059e-01
#-------------------------
#-------------------------
#     CI = 1/(2I + 1)
#-------------------------
#-------------------------
    if x >= -0.39 && x <= 0.57 # go to 100
        if x < -0.18 # go to 10
            u = (x + 0.3)/0.7
            up2 = u + 2.0
            w1 = a - u*0.3
        elseif x > 0.18 # go to 20
            t = 0.75*x
            u = t - 0.25
            up2 = t + 1.75
            w1 = b + u/3.0
        else
            u = x
            up2 = u + 2.0
            w1 = 0.0
        end
#
#                  SERIES EXPANSION
#
        r = u/up2
        t = r*r
#
#        Z IS A MINIMAX APPROXIMATION OF THE SERIES
#
#               C6 + C7*R**2 + C8*R**4 + ...
#
#        FOR THE INTERVAL (0.0, 0.375). THE APPROX-
#        IMATION IS ACCURATE TO WITHIN 1.6 UNITS OF
#        THE 21-ST SIGNIFICANT DIGIT.
#
      z = @horner(t,
                   .7692307692307692307680e-01,
                  -.1505958055914600184836e+00,
                   .9302355725278521726994e-01,
                  -.1787900022182327735804e-01) /
        @horner(t,1.0,
                -.2824412139355646910683e+01,
                 .2892424216041495392509e+01,
                -.1263560605948009364422e+01,
                 .1966769435894561313526e+00)
      w = @horner(t,
                  .333333333333333333333333333333333e+00,
                  .200000000000000000000000000000000e+00,
                  .142857142857142857142857142857143e+00,
                  .111111111111111111111111111111111e+00,
                  .909090909090909090909090909090909e-01,
                  z)

        return r*(2.0*t*w - u) - w1
    end
    return log1p(x) - x
end




# Stirling series for the gamma function
# 
# stirling(x) = gamma(x) * e^x / (x^(x-0.5) * √2π)
#             = 1 + 1/(12x) + 1/(288x^2) - 139/(51_840z^3) + ...

# TODO: create dedicated function, as working in
# log-space will lose a few bits of precision.
stirling(x) = exp(lstirling(x))

# lstirling(x) = log(stirling(x))
#              = lgamma(x) + x - (x-0.5)*log(x) - 0.5*log2π
#              = 1/(12x) - 1/(360x^3) + 1/(1260x^5) + ...

# fallback
lstirling(x) = lgamma(x)- (x-0.5)*log(x) + x - 0.5*oftype(x,log2π)
lstirling(x::Integer) = lstirling(float(x))

# based on NSWC DPDEL: only valid for values >= 10
# Float32 version?
function lstirling(x::Float64)
    if x <= 10.0
        return lgamma(x) - (x-0.5)*log(x) + x - 0.5*log2π
    else
        u = 10.0/x
        t = u*u
        return @horner(t,
                .833333333333333333333333333333e-01,
                -.277777777777777777777777752282e-04,
                .793650793650793650791732130419e-07,
                -.595238095238095232389839236182e-09,
                .841750841750832853294451671990e-11,
                -.191752691751854612334149171243e-12,
                .641025640510325475730918472625e-14,
                -.295506514125338232839867823991e-15,
                .179643716359402238723287696452e-16,
                -.139228964661627791231203060395e-17,
                .133802855014020915603275339093e-18,
                -.154246009867966094273710216533e-19,
                .197701992980957427278370133333e-20,
                -.234065664793997056856992426667e-21,
                .171348014966398575409015466667e-22) / x
    end
end

# The regularized incomplete gamma function
# Translated from the NSWC Library
function gratio(a::Float32, x::Float32, ind::Integer)
#-----------------------------------------------------------------------
#
#        EVALUATION OF THE INCOMPLETE GAMMA RATIO FUNCTIONS
#                      P(A,X) AND Q(A,X)
#
#                        ----------
#
#     IT IS ASSUMED THAT A AND X ARE NONNEGATIVE, WHERE A AND X
#     ARE NOT BOTH 0.
#
#     ANS AND QANS ARE VARIABLES. GRATIO ASSIGNS ANS THE VALUE
#     P(A,X) AND QANS THE VALUE Q(A,X). IND MAY BE ANY INTEGER.
#     IF IND = 0 THEN THE USER IS REQUESTING AS MUCH ACCURACY AS
#     POSSIBLE (UP TO 14 SIGNIFICANT DIGITS). OTHERWISE, IF
#     IND = 1 THEN ACCURACY IS REQUESTED TO WITHIN 1 UNIT OF THE
#     6-TH SIGNIFICANT DIGIT, AND IF IND .NE. 0,1 THEN ACCURACY
#     IS REQUESTED TO WITHIN 1 UNIT OF THE 3RD SIGNIFICANT DIGIT.
#
#     ERROR RETURN ...
#
#        ANS IS ASSIGNED THE VALUE 2 WHEN A OR X IS NEGATIVE,
#     WHEN A*X = 0, OR WHEN P(A,X) AND Q(A,X) ARE INDETERMINANT.
#     P(A,X) AND Q(A,X) ARE COMPUTATIONALLY INDETERMINANT WHEN
#     X IS EXCEEDINGLY CLOSE TO A AND A IS EXTREMELY LARGE.
#
#-----------------------------------------------------------------------
#     WRITTEN BY ALFRED H. MORRIS, JR.
#        NAVAL SURFACE WARFARE CENTER
#        DAHLGREN, VIRGINIA
#     REVISED ... DEC 1991
#-------------------------
#     REAL J, L, ACC0(3), BIG(3), E0(3), X0(3), WK(20)
    wk = Array(Float32, 20)
#     REAL A0(4), A1(4), A2(2), A3(2), A4(2), A5(2), A6(2), A7(2),
#    *     A8(2)
#     REAL B0(6), B1(4), B2(5), B3(5), B4(4), B5(3), B6(2), B7(2)
#     REAL D0(6), D1(4), D2(2), D3(2), D4(1), D5(1), D6(1)
#-------------------------
    acc0 = [5.f-15, 5.f-7, 5.f-4]
    big = [25.0f0, 14.0f0, 10.0f0]
    e0 = [.25f-3, .25f-1, .14f0]
    x0 = [31.0f0, 17.0f0, 9.7f0]
#-------------------------
#     ALOG10 = LN(10)
#     RT2PIN = 1/SQRT(2*PI)
#     RTPI   = SQRT(PI)
#-------------------------
    alog10 = 2.30258509299405f0
    rt2pin = .398942280401433f0
    rtpi   = 1.77245385090552f0
#-------------------------
#
#             COEFFICIENTS FOR MINIMAX APPROXIMATIONS
#                          FOR C0,...,C8
#
#-------------------------
#-------------------------
#
#     ****** E IS A MACHINE DEPENDENT CONSTANT. E IS THE SMALLEST
#            FLOATING POINT NUMBER FOR WHICH 1.0 + E .GT. 1.0 .
#
    e = eps(Float32)
#-------------------------
    if a < 0.0 || x < 0.0 throw(DomainError()) end # go to 400
    if a == 0.0 && x == 0.0 throw(DomainError()) end # go to 400
    if a*x == 0.0 # go to 331
        if x < a return 0.0f0, 1.0f0 end
        return 1.0f0, 0.0f0
    end

    iop = ind + 1
    if iop != 1 && iop != 2 iop = 3 end
    acc = max(acc0[iop],e)

#            SELECT THE APPROPRIATE ALGORITHM

    if a < 1.0 # go to 10
        if a == 0.5 # go to 320
            if x < 0.25 # go to 321
                ans = erf(sqrt(x))
                return ans, 0.5f0 + (0.5f0 - ans)
            end
            qans = erfc(sqrt(x))
            return 0.5f0 + (0.5f0 - qans), qans
        end
        if x < 1.1 # go to 110

#             TAYLOR SERIES FOR P(A,X)/X**A

            l = 3.0f0
            c = x
            sum = x/(a + 3.0f0)
            tol = 3.0f0*acc/(a + 1.0f0)
            while true
                l += 1.0f0
                c *= -(x/l)
                t = c/(a + l)
                sum += t
                if abs(t) <= tol break end
            end
            j = a*x*((sum/6.0f0 - 0.5f0/(a + 2.0f0))*x + 1.0f0/(a + 1.0f0))
    
            z = a*log(x)
            h = rgamma1pm1(a)
            g = 1.0f0 + h
            if x >= 0.25 # go to 120
                if a < x/2.59f0 # go to 135
                    l = expm1(z)
                    w = 0.5f0 + (0.5f0 + l)
                    qans = (w*j - l)*g - h
                    if qans < 0.0 return 1.0f0, 0.0f0 end # go to 310
                    return 0.5f0 + (0.5f0 - qans), qans
                end
                w = exp(z)
                ans = w*g*(0.5f0 + (0.5f0 - j))
                return ans, 0.5f0 + (0.5f0 - ans)
            end
            if z <= -.13394 # go to 135
                w = exp(z)
                ans = w*g*(0.5f0 + (0.5f0 - j))
                return ans, 0.5f0 + (0.5f0 - ans)
            end
            l = expm1(z)
            w = 0.5f0 + (0.5f0 + l)
            qans = (w*j - l)*g - h
            if qans < 0.0 return 1.0f0, 0.0f0 end # go to 310
            return 0.5f0 + (0.5f0 - qans), qans
        end
        r = rcomp(a, x)
        if r == 0.0 return 1.0f0, 0.0f0 end # go to 310

#              CONTINUED FRACTION EXPANSION

        tol = max(8.0f0*e,4.0f0*acc)
        a2nm1 = 1.0f0
        a2n = 1.0f0
        b2nm1 = x
        b2n = x + (1.0f0 - a)
        c = 1.0f0
        while true
            a2nm1 = x*a2n + c*a2nm1
            b2nm1 = x*b2n + c*b2nm1
            c += 1.0f0
            t = c - a
            a2n = a2nm1 + t*a2n
            b2n = b2nm1 + t*b2n

            a2nm1 /= b2n
            b2nm1 /= b2n
            a2n /= b2n
            b2n = 1.0f0
            if abs(a2n - a2nm1/b2nm1) < tol*a2n break end
        end

        qans = r*a2n
        return 0.5f0 + (0.5f0 - qans), qans
    end

    if a >= big[iop] # go to 20
        l = x/a
        if l == 0.0 return 0.0f0, 1.0f0 end # go to 300
        s = 0.5f0 + (0.5f0 - l)
        z = -logmxp1(l)
        if z >= 700.0f0/a # go to 330
            if abs(s) <= 2.0f0*e error() end # go to 400
            if x <= a return 0.0f0, 1.0f0 end # go to 300
            return 1.0f0, 0.0f0
        end
        y = a*z
        rta = sqrt(a)
        if abs(s) <= e0[iop]/rta # go to 250

#               TEMME EXPANSION FOR L = 1

            if a*e*e > 3.28f-3 error() end # go to 400
            c = 0.5f0 + (0.5f0 - y)
            w = (0.5f0 - sqrt(y)*(0.5f0 + (0.5f0 - y/3.0f0))/rtpi)/c
            u = 1.0f0/a
            z = sqrt(z + z)
            if l < 1.0 z = -z end
            if iop < 2 # 260,270,280
                c0 = @horner(z, -.333333333333333f+00, 
                                 .833333333333333f-01,
                                -.148148148148148f-01,
                                 .115740740740741f-02)
                c1 = @horner(z, -.185185185185185f-02, 
                                -.347222222222222f-02,
                                 .264550264550265f-02, 
                                -.990226337448560f-03)
                c2 = @horner(z,  .413359788359788f-02, 
                                -.268132716049383f-02,
                                 .771604938271605f-03)
                c3 = @horner(z,  .649434156378601f-03, 
                                 .229472093621399f-03,
                                -.469189494395256f-03)
                c4 = @horner(z, -.861888290916712f-03, 
                                 .784039221720067f-03)
                c5 = @horner(z, -.336798553366358f-03, 
                                -.697281375836586f-04)
                c6 = @horner(z,  .531307936463992f-03, 
                                -.592166437353694f-03)
                t  = (((((((-.652623918595309f-03*u + .344367606892378f-03)*u + c6)*u + c5)*u + c4)*u + c3)*u + c2)*u + c1)*u + c0
            elseif iop == 2
                c0 = @horner(z, -.333333333333333f+00,
                                 .833333333333333f-01,
                                -.148148148148148f-01)
                c1 = @horner(z, -.185185185185185f-02, 
                                -.347222222222222f-02)
                t  = (d20*u + c1)*u + c0
            else
                t  = @horner(z, -.333333333333333f+00, 
                                 .833333333333333f-01)
            end # go to 240
            if l >= 1.0 # go to 241
                qans = c*(w + rt2pin*t/rta)
                return 0.5f0 + (0.5f0 - qans), qans
            end
            ans = c*(w - rt2pin*t/rta)
            return ans, 0.5f0 + (0.5f0 - ans)
        end
        if abs(s) <= 0.4 # go to 200
            if abs(s) <= 2.0f0*e && a*e*e > 3.28e-3 error() end # go to 400
            c = exp(-y)
            w = 0.5f0*erfcx(sqrt(y))
            u = 1.0f0/a
            z = sqrt(z + z)
            if l < 1.0 z = -z end
            if iop < 2 # 210,220,230

                if abs(s) <= 1.e-3 # go to 260
                    c0 = @horner(z, -.333333333333333f+00, 
                                     .833333333333333f-01,
                                    -.148148148148148f-01,
                                     .115740740740741f-02)
                    c1 = @horner(z, -.185185185185185f-02, 
                                    -.347222222222222f-02,
                                     .264550264550265f-02, 
                                    -.990226337448560f-03)
                    c2 = @horner(z,  .413359788359788f-02, 
                                    -.268132716049383f-02,
                                     .771604938271605f-03)
                    c3 = @horner(z,  .649434156378601f-03, 
                                     .229472093621399f-03,
                                    -.469189494395256f-03)
                    c4 = @horner(z, -.861888290916712f-03, 
                                     .784039221720067f-03)
                    c5 = @horner(z, -.336798553366358f-03, 
                                    -.697281375836586f-04)
                    c6 = @horner(z,  .531307936463992f-03, 
                                    -.592166437353694f-03)
                    t  = (((((((-.652623918595309f-03*u + .344367606892378f-03)*u + c6)*u + c5)*u + c4)*u + c3)*u + c2)*u + c1)*u + c0
                else

#            using the minimax approximations

                    c0 = @horner(z,-.333333333333333f+00,
                                   -.159840143443990f+00,
                                   -.335378520024220f-01,
                                   -.231272501940775f-02)/
                         @horner(z,1.0f0,
                                    .729520430331981f+00,
                                    .238549219145773f+00,
                                    .376245718289389f-01,
                                    .239521354917408f-02,
                                   -.939001940478355f-05,
                                    .633763414209504f-06)
                    c1 = @horner(z,-.185185185184291f-02,
                                   -.491687131726920f-02,
                                   -.587926036018402f-03,
                                   -.398783924370770f-05)/
                         @horner(z,1.0f0,
                                    .780110511677243f+00,
                                    .283344278023803f+00,
                                    .506042559238939f-01,
                                    .386325038602125f-02)
                    c2 = @horner(z, .413359788442192f-02,
                                    .669564126155663f-03)/
                         @horner(z,1.0f0,
                                    .810647620703045f+00,
                                    .339173452092224f+00,
                                    .682034997401259f-01,
                                    .650837693041777f-02,
                                    .421924263980656f-03)
                    c3 = @horner(z, .649434157619770f-03,
                                    .810586158563431f-03)/
                         @horner(z,1.0f0,
                                    .894800593794972f+00,
                                    .406288930253881f+00,
                                    .906610359762969f-01,
                                    .905375887385478f-02,
                                   -.632276587352120f-03)
                    c4 = @horner(z,-.861888301199388f-03,
                                   -.105014537920131f-03)/
                         @horner(z,1.0f0,
                                    .103151890792185f+01,
                                    .591353097931237f+00,
                                    .178295773562970f+00,
                                    .322609381345173f-01)
                    c5 = @horner(z,-.336806989710598f-03,
                                   -.435211415445014f-03)/
                         @horner(z,1.0f0,
                                    .108515217314415f+01,
                                    .600380376956324f+00,
                                    .178716720452422f+00)
                    c6 = @horner(z, .531279816209452f-03,
                                   -.182503596367782f-03)/
                         @horner(z,1.0f0,
                                    .770341682526774f+00,
                                    .345608222411837f+00)
                    c7 = @horner(z, .344430064306926f-03,
                                    .443219646726422f-03)/
                         @horner(z,1.0f0,
                                    .115029088777769f+01,
                                    .821824741357866f+00)
                    c8 = .878371203603888f-03*z - .686013280418038f-03
                    t = (((((((c8*u + c7)*u + c6)*u + c5)*u + c4)*u + c3)*u + c2)*u + c1)*u + c0
                end
            elseif iop == 2

#                    temme expansion
    
                c0 = @horner(z, -.333333333333333f+00, 
                                 .833333333333333f-01,
                                -.148148148148148f-01, 
                                 .115740740740741f-02,
                                 .352733686067019f-03,
                                -.178755144032922f-03,
                                 .391926317852244f-04)
                c1 = @horner(z, -.185185185185185f-02, 
                                -.347222222222222f-02,
                                 .264550264550265f-02,
                                -.990226337448560f-03,
                                 .205761316872428f-03)
                c2 = @horner(z,  .413359788359788f-02, 
                                -.268132716049383f-02)
                t  = (c2*u + c1)*u + c0
            else
                t  = @horner(z, -.333333333333333f+00, 
                                 .833333333333333f-01,
                                -.148148148148148f-01, 
                                 .115740740740741f-02)
            end
            if l >= 1.0 # go to 241
                qans = c*(w + rt2pin*t/rta)
                return 0.5f0 + (0.5f0 - qans), qans
            end
            ans = c*(w - rt2pin*t/rta)
            return ans, 0.5f0 + (0.5f0 - ans)
        end
    end
    if a <= x && x < x0[iop] # go to 30
        twoa = a + a
        m = int(twoa)
        if twoa == m # go to 30
            i = div(m,2)
            if a == i # go to 140

#             FINITE SUMS FOR Q WHEN A .GE. 1
#                 AND 2*A IS AN INTEGER

                sum = exp(-x)
                t = sum
                n = 1
                c = 0.0f0
            else
                rtx = sqrt(x)
                sum = erfc(rtx)
                t = exp(-x)/(rtpi*rtx)
                n = 0
                c = -0.5f0
            end
            while n != i # go to 161
                n += 1
                c += 1.0f0
                t = (x*t)/c
                sum += t
            end
            return 0.5f0 + (0.5f0 - sum), sum
        end
    end

    r = rcomp(a, x)
    if r == 0.0 # go to 331
        if x < a return 0.0f0, 1.0f0 end
        return 0.0f0, 1.0f0
    end
    if x > max(a, alog10) # go to 50
        if x < x0[iop] # go to 170

#              CONTINUED FRACTION EXPANSION

            tol = max(8.0f0*e,4.0f0*acc)
            a2nm1 = 1.0f0
            a2n = 1.0f0
            b2nm1 = x
            b2n = x + (1.0f0 - a)
            c = 1.0f0
            while true
                a2nm1 = x*a2n + c*a2nm1
                b2nm1 = x*b2n + c*b2nm1
                c += 1.0f0
                t = c - a
                a2n = a2nm1 + t*a2n
                b2n = b2nm1 + t*b2n
    
                a2nm1 /= b2n
                b2nm1 /= b2n
                a2n /= b2n
                b2n = 1.0f0
                if abs(a2n - a2nm1/b2nm1) < tol*a2n break end
            end
    
            qans = r*a2n
            return 0.5f0 + (0.5f0 - qans), qans
        end
      
#                 ASYMPTOTIC EXPANSION

        amn = a - 1.0f0
        t = amn/x
        wk[1] = t
        n = 0
        for n = 2:20
            amn -= 1.0f0
            t *= amn/x
            if abs(t) <= 1.f-3 break end # go to 90
            wk[n] = t
        end
        sum = t
        while abs(t) >= acc # go to 100
            amn -= 1.0f0
            t *= amn/x
            sum += t
        end

        mx = n - 1
        for m = 1:mx
            n -= 1
            sum += wk[n]
        end
        qans = (r/x)*(1.0f0 + sum)
        return 0.5f0 + (0.5f0 - qans), qans
    end

#                 TAYLOR SERIES FOR P/R

    apn = a + 1.0f0
    t = x/apn
    wk[1] = t
    n = 0
    for n = 2:20
        apn += 1.0f0
        t *= x/apn
        if t <= 1.f-3 break end # go to 60
        wk[n] = t
    end
    sum = t
    tol = 0.5f0*acc
    while true
        apn += 1.0f0
        t *= x/apn
        sum += t
        if t <= tol break end # go to 61
    end

    mx = n - 1
    for m = 1:mx
        n -= 1
        sum += wk[n]
    end
    ans = (r/a)*(1.0f0 + sum)
    return ans, 0.5f0 + (0.5f0 - ans)
end

function dgrat(a::Real, x::Real)
#-----------------------------------------------------------------------
#
#        EVALUATION OF THE INCOMPLETE GAMMA RATIO FUNCTIONS
#                      P(A,X) AND Q(A,X)
#
#-----------------------------------------------------------------------
#     WRITTEN BY ALFRED H. MORRIS, JR.
#        NAVAL SURFACE WARFARE CENTER
#        DAHLGREN, VIRGINIA
#     REVISED ... JAN 1992
#-------------------------
#     DOUBLE PRECISION A, X, ANS, QANS
#     DOUBLE PRECISION AMN, ALOG10, APN, A2N, A2NM1, BIG, B2N,
#    *         B2NM1, C, E, G, H, J, L, R, RTA, RTPI, RTX, S,
#    *         SUM, T, TOL, TWOA, U, X0, Y, Z, WK(20)
    wk = Array(Float64, 20)
#     DOUBLE PRECISION DPMPAR, DRLOG, DREXP
#     DOUBLE PRECISION DERF, DERFC1, DGAM1, DRCOMP
#-------------------------
#     ALOG10 = LN(10)
#     RTPI   = DSQRT(PI)
#-------------------------
    alog10  = 2.30258509299404568401799145468e0
    rtpi    = 1.77245385090551602729816748334e0
#-------------------------
#
#     ****** E IS A MACHINE DEPENDENT CONSTANT. E IS THE SMALLEST
#            FLOATING POINT NUMBER FOR WHICH 1.0 + E .GT. 1.0 .
#
    e = eps(typeof(float(x)))
    n = 0
    s = 1.0
    r = 0.0
#
#-------------------------
    if a < 0.0 || x < 0.0 throw(DomainError()) end
    if a == 0.0 && x == 0.0 throw(DomainError()) end
    if a*x == 0.0 # go to 331
        if x <= a return 0.0, 1.0 end
        return 1.0, 0.0
    end

    e = max(e,1.e-30)
    if e < 1.e-17 
        big = 50.0 
        x0 = 68.0
    else
        big = 30.0
        x0 = 45.0
    end
    

#            SELECT THE APPROPRIATE ALGORITHM
    while true
        if a < 1.0 # go to 10
            if a == 0.5 # go to 320
                if x < 0.25 # go to 321
                    ans = erf(sqrt(x))
                    return ans, 0.5 + (0.5 - ans)
                end
                qans = erfc(sqrt(x))
                return 0.5 + (0.5 - qans), qans
            end
            if x <= 2.0 # go to 110
            # taylor series for p(a,x)/x**a
                l = 3.0
                c = x
                sum = x/(a + 3.0)
                tol = 3.0*e/(a + 1.0)
                while true
                    l += 1.0
                    c *= -(x/l)
                    t = c/(a + l)
                    sum += t
                    if abs(t) <= tol break end
                end
                j = a*x*((sum/6.0 - 0.5/(a + 2.0))*x + 1.0/(a + 1.0))
    
                z = a*log(x)
                u = exp(z)
                h = rgamma1pm1(a)
                g = 1.0 + h
                ans = u*g*(0.5 + (0.5 - j))
                qans = 0.5 + (0.5 - ans)
                if ans <= 0.9 return ans, qans end
    
                l = expm1(z)
                qans = (u*j - l)*g - h
                if qans <= 0.0 return 1.0, 0.0 end
                return 0.5 + (0.5 - qans), qans
            end
            r = drcomp(a,x)
            if r == 0.0 return 1.0, 0.0 end
            break
        end # go to 170
        while true
            if a < big # go to 20
                if a > x || x >= x0 break end# go to 30
                twoa = a + a
                m = itrunc(twoa)
                l = float(m)
                if twoa != l break end # go to 30
                i = div(m,2)
                l = float(i)
                if a == l # go to 140
                    sum = exp(-x)
                    t = sum
                    n = 1
                    c = 0.0
                else
                    rtx = sqrt(x)
                    sum = erfc(rtx)
                    t = exp(-x)/(rtpi*rtx)
                    n = 0
                    c = -0.5
                end
                while n != i # go to 161
                    n += 1
                    c += 1.0
                    t = (x*t)/c
                    sum += t
                end
                qans = sum
                return 0.5 + (0.5 - qans), qans
            end
            
            l = x/a
            if l == 0.0 return 0.0, 1.0 end
            s = 0.5 + (0.5 - l)
            z = -logmxp1(l)
            if z >= 700.0/a # go to 330
                if abs(s) <= 2.0*e error("ierr=3") end
                if x < a return 0.0, 1.0 end
                return 1.0, 0.0
            end
            y = a*z
            rta = sqrt(a)
            if abs(s) <= 0.4 # go to 200
                if abs(s) <= 2.0*e && a*e*e > 3.28e-3 error("ierr=3") end
                if e <= 1.e-17 return dgr29(a, y, l, z, rta) end
                return dgr17(a, y, l, z, rta)
            end
            break
        end

        r = drcomp(a,x)
        if r == 0.0 # go to 331
            if abs(s) <= 2.0*e error("ierr=3") end
            if x < a return 0.0, 1.0 end
            return 1.0, 0.0
        end
        if x > max(a,alog10) # go to 50
            if x < x0 break end# go to 170
        else # go to 80

                  # TAYLOR SERIES FOR P/R

            apn = a + 1.0
            t = x/apn
            wk[1] = t
            for n = 2:20
                apn += 1.0
                t *= x/apn
                if t < 1.e-3 break end # go to 60
                wk[n] = t
            end
    
            sum = t
            tol = 0.5*e
            while true
                apn += 1.0
                t *= x/apn
                sum += t
                if t <= tol break end # go to 61
            end
    
            mx = n - 1
            for m = 1:mx
                n -= 1
                sum += wk[n]
            end
            ans = (r/a)*(1.0 + sum)
            return ans, 0.5 + (0.5 - ans)
        end
    
              # ASYMPTOTIC EXPANS
        amn = a - 1.0
        t = amn/x
        wk[1] = t
        for n = 2:20
            amn -= 1.0
            t *= amn/x
            if abs(t) <= 1.e-3 break end # go to 90
            wk[n] = t
        end

        sum = t
        while abs(t) >= e # go to 100
            amn -= 1.0
            t *= amn/x
            sum += t
        end

        mx = n - 1
        for m = 1:mx
            n -= 1
            sum += wk[n]
        end
        qans = (r/x)*(1.0 + sum)
        return 0.5 + (0.5 - qans), qans
    end

#              continued fraction expansion

    tol = 8.0*e
    a2nm1 = 1.0
    a2n = 1.0
    b2nm1 = x
    b2n = x + (1.0 - a)
    c = 1.0
    while true
        a2nm1 = x*a2n + c*a2nm1
        b2nm1 = x*b2n + c*b2nm1
        c += 1.0
        t = c - a
        a2n = a2nm1 + t*a2n
        b2n = b2nm1 + t*b2n
        a2nm1 = a2nm1/b2n
        b2nm1 = b2nm1/b2n
        a2n = a2n/b2n
        b2n = 1.0
        if abs(a2n - a2nm1/b2nm1) < tol*a2n break end # go to 180
    end
    qans = r*a2n
    return 0.5 + (0.5 - qans), qans
end

function rcomp(a::Float32, x::Float32)
#-----------------------------------------------------------------------
#                EVALUATION OF EXP(-X)*X**A/GAMMA(A)
#-----------------------------------------------------------------------
#     RT2PIN = 1/SQRT(2*PI)
#------------------------
    rt2pin = .398942280401433f0
#------------------------
    if x == 0.0 return 0.0f0 end
    if a < 20.0 # go to 20

        t = a*log(x) - x
        if t < realminexp(Float32) return 0.0f0 end
        if a < 1.0 # go to 10
            return (a*exp(t))*(1.0f0 + rgamma1pm1(a))
        end
        return exp(t)/gamma(a)
    end

    u = x/a
    if u == 0.0 return 0.0f0 end
    t = (1.0f0/a)^2
    t1 = (((0.75f0*t - 1.0f0)*t + 3.5f0)*t - 105.0f0)/(a*1260.0f0)
    t1 += a*logmxp1(u)
    if t1 >= realminexp(Float32) return rt2pin*sqrt(a)*exp(t1) end
end

function drcomp(a::Real, x::Real)
#-----------------------------------------------------------------------
#              EVALUATION OF EXP(-X)*X**A/GAMMA(A)
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A, X, C, T, W
#     DOUBLE PRECISION DGAMMA, DGAM1, DPDEL, DRLOG, DXPARG
#--------------------------
#     C = 1/SQRT(2*PI)
#--------------------------
    c = .398942280401432677939946059934
#--------------------------
    if x == 0.0 return 0.0 end
    if a <= 20.0 # go to 20
        t = a*log(x) - x
        if t < realminexp(Float64) return 0.0 end
        if a < 1.0 # go to 10
            return (a*exp(t))*(1.0 + rgamma1pm1(a))
        end
        return exp(t)/gamma(a)
    end

    t = x/a
    if t == 0.0 return 0.0 end
    w = -(lstirling(a) - a*logmxp1(t))
    if w >= realminexp(Float64) 
        return c*sqrt(a)*exp(w)
    else
        return 0.0
    end
end



function dgr17(a::Real, y::Real, l::Real, z::Real, rta::Real)
#-----------------------------------------------------------------------
#
#            ALGORITHM USING MINIMAX APPROXIMATIONS
#                        FOR C0,...,C10
#
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A, Y, L, Z, RTA, ANS, QANS
#     DOUBLE PRECISION E, RT2PIN, T, U, W
#     DOUBLE PRECISION C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10
#     DOUBLE PRECISION A0(5), A1(4), A2(4), A3(4), A4(3), A5(2),
#    *                 A6(2), A7(2), A8(2), A9(3), A10(2)
#     DOUBLE PRECISION B0(6), B1(6), B2(5), B3(4), B4(4), B5(4),
#    *                 B6(3), B7(3), B8(2)
#     DOUBLE PRECISION DERFC1
#------------------------
#     RT2PIN = 1/DSQRT(2*PI)
#------------------------
    rt2pin = .398942280401432678e0
#------------------------
    e = exp(-y)
    w = 0.5*erfcx(sqrt(y))
    u = 1.0/a
    z = sqrt(z + z)
    if l < 1.0 z = -z end

    c0 = @horner(z,  -.33333333333333333e+00,
                               -.24232172943558393e+00,
                               -.76816029947195974e-01,
                               -.11758531313175796e-01,
                               -.73324404807556026e-03)
    c0 /= @horner(z,                     1.0,
                                .97696518830675185e+00, 
                                .43024494247383254e+00,
                                .10288837674434487e+00, 
                                .13250270182342259e-01,
                                .73121701584237188e-03, 
                                .10555647473018528e-06)
    c1 = @horner(z,  -.18518518518518417e-02,
                               -.52949366601406939e-02,
                               -.16090334014223031e-02,
                               -.16746784557475121e-03)
    c1 /= @horner(z, 1.0,
                                .98426579647613593e+00,
                                .45195109694529839e+00,
                                .11439610256504704e+00,
                                .15954049115266936e-01,
                                .98671953445602142e-03,
                                .12328086517283227e-05)
    c2 = @horner(z,   .41335978835983393e-02,
                                .15067356806896441e-02,
                                .13743853858711134e-03,
                                .12049855113125238e-04)
    c2 /= @horner(z, 1.0,
                                .10131761625405203e+01,
                                .50379606871703058e+00,
                                .14009848931638062e+00,
                                .22316881460606523e-01,
                                .15927093345670077e-02)
    c3 = @horner(z,   .64943415637082551e-03,
                                .81804333975935872e-03,
                                .13012396979747783e-04,
                                .46318872971699924e-05)
    c3 /= @horner(z, 1.0,
                                .90628317147366376e+00,
                                .42226789458984594e+00,
                                .10044290377295469e+00,
                                .12414068921653593e-01)
    c4 = @horner(z,  -.86188829773520181e-03,
                               -.82794205648271314e-04,
                               -.37567394580525597e-05)
    c4 /= @horner(z, 1.0,
                                .10057375981227881e+01,
                                .57225859400072754e+00,
                                .16988291247058802e+00,
                                .31290397554562032e-01)
    c5 = @horner(z,  -.33679854644784478e-03,
                               -.43263341886764011e-03)
    c5 /= @horner(z, 1.0,
                                .10775200414676195e+01,
                                .60019022026983067e+00,
                                .17081504060220639e+00,
                                .22714615451529335e-01)

    c6 = @horner(z,   .53130115408837152e-03,
                               -.12962670089753501e-03)
    c6 /= @horner(z, 1.0,
                                .87058903334443855e+00,
                                .45957439582639129e+00,
                                .65929776650152292e-01)
    c7 = @horner(z,   .34438428473168988e-03,
                                .47861364421780889e-03)
    c7 /= @horner(z, 1.0,
                                .12396875725833093e+01,
                                .78991370162247144e+00,
                                .27176241899664174e+00)
    c8 = @horner(z,  -.65256615574219131e-03,
                                .27086391808339115e-03)
    c8 /= @horner(z, 1.0,
                                .87002402612484571e+00,
                                .44207055629598579e+00)
    c9 = @horner(z,  -.60335050249571475e-03,
                               -.14838721516118744e-03,
                                .84725086921921823e-03)
    c10 = @horner(z,  .13324454494800656e-02,
                               -.19144384985654775e-02)


    t = (((((((((c10*u + c9)*u + c8)*u + c7)*u + c6)*u + c5)*u + c4)*u + c3)*u + c2)*u + c1)*u + c0

    if (l >= 1.0) # go to 10
        qans = e*(w + rt2pin*t/rta)
        return 0.5 + (0.5 - qans), qans
    end
    ans = e*(w - rt2pin*t/rta)
    return ans, 0.5 + (0.5 - ans)
end

function dgr29(a::Real, y::Real, l::Real, z::Real, rta::Real)
#-----------------------------------------------------------------------
#
#            ALGORITHM USING MINIMAX APPROXIMATIONS
#
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A, Y, L, Z, RTA, ANS, QANS
#     DOUBLE PRECISION A0(7), A1(7), A2(7), A3(7), A4(7), A5(7), A6(4),
#    *                 A7(5), A8(5), A9(5), A10(4), A11(4), A12(4),
#    *                 A13(3), A14(3), A15(2), A16(2), A17(1), A18(1)
#     DOUBLE PRECISION B0(9), B1(9), B2(8), B3(8), B4(8), B5(7), B6(9),
#    *                 B7(7), B8(7), B9(6), B10(6), B11(5), B12(4),
#    *                 B13(4), B14(2), B15(2), B16(1)
#     DOUBLE PRECISION C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10,
#    *                 C11, C12, C13, C14, C15, C16
#     DOUBLE PRECISION D0(7), E, RT2PIN, T, U, W
#     DOUBLE PRECISION DERFC1
#---------------------------
#     RT2PIN = 1/DSQRT(2*PI)
#---------------------------
    rt2pin = .398942280401432677939946059934e0
#---------------------------

    e = exp(-y)
    w = 0.5*erfcx(sqrt(y))
    u = 1.0/a
    z = sqrt(z + z)
    if l < 1.0 z = -z end

    t = @horner(z,-.218544851067999216147364227e-05,
                            -.490033281596113358850307112e-05,
                            -.372722892959910688597417881e-05,
                            -.145717031728609218851588740e-05,
                            -.327874000161065050049103731e-06,
                            -.408902435641223939887180303e-07,
                            -.234443848930188413698825870e-08)
    t /= @horner(z,1.0, 
                             .139388806936391316154237713e+01,
                             .902581259032419042347458484e+00,
                             .349373447613102956696810725e+00,
                             .866750030433403450681521877e-01,
                             .138263099503103838517015533e-01,
                             .131659965062389880196860991e-02,
                             .597739416777031660496708557e-04,
                             .319268409139858531586963150e-08,
                            -.129786815987713980865910767e-09)
    c0 = ((((((t*z +       .391926317852243778169704095630e-04)*z + 
                          -.178755144032921810699588477366e-03)*z + 
                           .352733686067019400352733686067e-03)*z + 
                           .115740740740740740740740740741e-02)*z +
                          -.148148148148148148148148148148e-01)*z + 
                           .833333333333333333333333333333e-01)*z + 
                          -.333333333333333333333333333333e+00
    c1 = @horner(z,-.185185185185185185185185200e-02,
                             -.627269388216833251971110268e-02,
                             -.462960105006279850867332060e-02,
                             -.167787748352827199882047653e-02,
                             -.334816794629374699945489443e-03,
                             -.359791514993122440319624428e-04,
                             -.162671127226300802902860047e-05)
    c1 /= @horner(z,1.0,
                               .151225469637089956064399494e+01,
                               .109307843990990308990473663e+01,
                               .482173396010404307346794795e+00,
                               .140741499324744724262767201e+00,
                               .276755209895072417713430394e-01,
                               .356903970692700621824901511e-02,
                               .275463718595762102271929980e-03,
                               .974094440943696092434381137e-05,
                               .361538770500640888027927000e-09)
    c2 = @horner(z,  .413359788359788359788359644e-02,
                               .365985331203490698463644329e-02,
                               .138385867950361368914038461e-02,
                               .287368655528567495658887760e-03,
                               .351658023234640143803014403e-04,
                               .261809837060522545971782889e-05,
                               .100841467329617467204527243e-06)
    c2 /= @horner(z,1.0,
                               .153405837991415136438992306e+01,
                               .114320896084982707537755002e+01,
                               .524238095721639512312120765e+00,
                               .160392471625881407829191009e+00,
                               .333036784835643463383606186e-01,
                               .457258679387716305283282667e-02,
                               .378705615967233119938297206e-03,
                               .144996224602847932479320241e-04)
    c3 = @horner(z,  .649434156378600823045102236e-03,
                               .141844584435355290321010006e-02,
                               .987931909328964685388525477e-03,
                               .331552280167649130371474456e-03,
                               .620467118988901865955998784e-04,
                               .695396758348887902366951353e-05,
                               .352304123782956092061364635e-06)
    c3 /= @horner(z,1.0,
                               .183078413578083710405050462e+01,
                               .159678625605457556492814589e+01,
                               .856743428738899911100227393e+00,
                               .308149284260387354956024487e+00,
                               .760733201461716525855765749e-01,
                               .126418031281256648240652355e-01,
                               .130398975231883219976260776e-02,
                               .656342109234806261144233394e-04)
    c4 = @horner(z, -.861888290916711698604710684e-03,
                              -.619343030286408407629007048e-03,
                              -.173138093150706317400323103e-03,
                              -.337525643163070607393381432e-04,
                              -.487392507564453824976295590e-05,
                              -.470448694272734954500324169e-06,
                              -.260879135093022176005540138e-07)
    c4 /= @horner(z,1.0,
                               .162826466816694512158165085e+01,
                               .133507902144433100426436242e+01,
                               .686949677014349678482109368e+00,
                               .241580582651643837306299024e+00,
                               .590964360473404599955095091e-01,
                               .990129468337836044520381371e-02,
                               .104553622856827932853059322e-02,
                               .561738585657138771286755470e-04)
    c5 = @horner(z, -.336798553366358151161633777e-03,
                              -.548868487607991087508092013e-03,
                              -.171902547619915856635305717e-03,
                              -.332229941748769925615918550e-04,
                              -.556701576804390213081214801e-05,
                               .506465072067030007394288471e-08,
                              -.116166342948098688243985652e-07)
    c5 /= @horner(z,1.0,
                               .142263185288429590449288300e+01,
                               .103913867517817784825064299e+01,
                               .462890328922621047510807887e+00,
                               .136071713023783507468096673e+00,
                               .254669201041872409738119341e-01,
                               .280714123386276098548285440e-02,
                               .106576106868815233442641444e-03)
    c6 = @horner(z,  .531307936463992224884286210e-03,
                               .209213745619758030399432459e-03,
                               .694345283181981060040314140e-05,
                               .118384620224413424936260301e-04)
    c6 /= @horner(z,1.0,
                               .150831585220968267709550582e+01,
                               .118432122801495778365352945e+01,
                               .571784440733980642101712125e+00,
                               .184699876959596092801262547e+00,
                               .384410125775084107229541456e-01,
                               .477475914272399601740818883e-02,
                               .151734058829700925162000373e-03,
                              -.248639208901374031411609873e-04,
                              -.633002360430352916354621750e-05)
    c7 = @horner(z,  .344367606892381545765962366e-03,
                               .605983804794748515383615779e-03,
                               .208913588225005764102252127e-03,
                               .462793722775687016808279009e-04,
                               .972342656522493967167788395e-05)
    c7 /= @horner(z,1.0,
                               .160951809815647533045690195e+01,
                               .133753662990343866552766613e+01,
                               .682159830165959997577293001e+00,
                               .230812334251394761909158355e+00,
                               .497403555098433701440032746e-01,
                               .621296161441756044580440529e-02,
                               .215964480325937088444595990e-03)
    c8 = @horner(z, -.652623918595320914510590273e-03,
                              -.353272052089782073130912603e-03,
                              -.282551884312564905942488077e-04,
                              -.192877995065652524742879002e-04,
                              -.231069438570167401077137510e-05)
    c8 /= @horner(z,1.0,
                               .182765408802230546887514255e+01,
                               .172269407630659768618234623e+01,
                               .101702505946784412105505734e+01,
                               .407929996207245634766606879e+00,
                               .110127834209242088316741250e+00,
                               .189231675289329563916597032e-01,
                               .156052480203446255774109882e-02)
    c9 = @horner(z, -.596761290192642722092337263e-03,
                              -.109151697941931403194363814e-02,
                              -.377126645910917006921076652e-03,
                              -.120148495117517992204095691e-03,
                              -.203007139532451428594124139e-04)
    c9 /= @horner(z,1.0,
                               .170833470935668756293234818e+01,
                               .156222230858412078350692234e+01,
                               .881575022436158946373557744e+00,
                               .335555306170768573903990019e+00,
                               .803149717787956717154553908e-01,
                               .108808775028021530146610124e-01)
    c10 =@horner(z,  .133244544950730832649306319e-02,
                               .580375987713106460207815603e-03,
                              -.352503880413640910997936559e-04,
                               .475862254251166503473724173e-04)
    c10/=@horner(z, 1.0,
                               .187235769169449339141968881e+01,
                               .183146436130501918547134176e+01,
                               .110810715319704031415255670e+01,
                               .448280675300097555552484502e+00,
                               .114651544043625219459951640e+00,
                               .161103572271541189817119144e-01)
    c11 =@horner(z,  .157972766214718575927904484e-02,
                               .246371734409638623215800502e-02,
                               .717725173388339108430635016e-05,
                               .121185049262809526794966703e-03)
    c11/=@horner(z, 1.0,
                               .145670749780693850410866175e+01,
                               .116082103318559904744144217e+01,
                               .505939635317477779328000706e+00,
                               .131627017265860324219513170e+00,
                               .794610889405176143379963912e-02)
    c12 =@horner(z, -.407251199495291398243480255e-02,
                              -.214376520139497301154749750e-03,
                               .650624975008642297405944869e-03,
                              -.246294151509758620837749269e-03)
    c12/=@horner(z, 1.0,
                               .162497775209192630951344224e+01,
                               .140298208333879535577602171e+01,
                               .653453590771198550320727688e+00,
                               .168390445944818504703640731e+00)
    c13 =@horner(z, -.594758070915055362667114240e-02,
                              -.109727312966041723997078734e-01,
                              -.159520095187034545391135461e-02)
    c13/=@horner(z, 1.0,
                               .175409273929961597148916309e+01,
                               .158706682625067673596619095e+01,
                               .790935125477975506817064616e+00,
                               .207815761771742289849225339e+00)
    c14 =@horner(z,  .175722793448246103440764372e-01,
                              -.119636668153843644820445054e-01,
                               .245543970647383469794050102e-02)
    c14/=@horner(z, 1.0,
                               .100158659226079685399214158e+01,
                               .676925518749829493412063599e+00)
    c15 =@horner(z,  .400765463491067514929787780e-01,
                               .588261033368548917447688791e-01)
    c15/=@horner(z, 1.0,
                               .149189509890654955611528542e+01,
                               .124266359850901469771032599e+01)
    c16 =                     (.119522261141925960204472459e+00*z + 
                              -.100326700196947262548667584e+00) / 
                              (.536462039767059451769400255e+00*z + 1.0)

    t =                       (.724036968309299822373280436e+00*u + 
                              -.259949826752497731336860753e+00)*u + c16
    t = (((((((((((((((t*u + c15)*u + c14)*u + c13)*u + c12)*u + c11)*u + c10)*u + c9)*u + c8)*u + c7)*u + c6)*u +
c5)*u + c4)*u + c3)*u + c2)*u + c1)*u + c0

    if l >= 1.0 # go to 10
        qans = e*(w + rt2pin*t/rta)
        return 0.5 + (0.5 - qans), qans
    end
    ans = e*(w - rt2pin*t/rta)
    return ans, 0.5 + (0.5 - ans)
end

# The inverse incomplete Gamma ratio function
# Translated from NSWC
function dginv(a::Real, p::Real, q::Real)
#-----------------------------------------------------------------------
#
#                        DOUBLE PRECISION
#             INVERSE INCOMPLETE GAMMA RATIO FUNCTION
#
#     GIVEN POSITIVE A, AND NONEGATIVE P AND Q WHERE P + Q = 1.
#     THEN X IS COMPUTED WHERE P(A,X) = P AND Q(A,X) = Q. SCHRODER
#     ITERATION IS EMPLOYED.
#
#                        ------------
#
#     X IS A VARIABLE. IF P = 0 THEN X IS ASSIGNED THE VALUE 0,
#     AND IF Q = 0 THEN X IS SET TO THE LARGEST FLOATING POINT
#     NUMBER AVAILABLE. OTHERWISE, DGINV ATTEMPTS TO OBTAIN
#     A SOLUTION FOR P(A,X) = P AND Q(A,X) = Q. IF THE ROUTINE
#     IS SUCCESSFUL THEN THE SOLUTION IS STORED IN X.
#
#     IERR IS A VARIABLE THAT REPORTS THE STATUS OF THE RESULTS.
#     WHEN THE ROUTINE TERMINATES, IERR HAS ONE OF THE FOLLOWING
#     VALUES ...
#
#       IERR =  0    THE SOLUTION WAS OBTAINED. ITERATION WAS
#                    NOT USED.
#       IERR.GT.0    THE SOLUTION WAS OBTAINED. IERR ITERATIONS
#                    WERE PERFORMED.
#       IERR = -2    (INPUT ERROR) A .LE. 0
#       IERR = -3    NO SOLUTION WAS OBTAINED. THE RATIO Q/A
#                    IS TOO LARGE.
#       IERR = -4    (INPUT ERROR) P OR Q IS NEGATIVE, OR
#                    P + Q .NE. 1.
#       IERR = -6    10 ITERATIONS WERE PERFORMED. THE MOST
#                    RECENT VALUE OBTAINED FOR X IS GIVEN.
#                    (THIS SETTING SHOULD NEVER OCCUR.)
#       IERR = -7    ITERATION FAILED. NO VALUE IS GIVEN FOR X.
#                    THIS MAY OCCUR WHEN X IS APPROXIMATELY 0.
#       IERR = -8    A VALUE FOR X HAS BEEN OBTAINED, BUT THE
#                    ROUTINE IS NOT CERTAIN OF ITS ACCURACY.
#                    ITERATION CANNOT BE PERFORMED IN THIS
#                    CASE. THIS SETTING CAN OCCUR ONLY WHEN
#                    P OR Q IS APPROXIMATELY 0.
#
#-----------------------------------------------------------------------
#     WRITTEN BY ALFRED H. MORRIS, JR.
#        NAVAL SURFACE WARFARE CENTER
#        DAHLGREN, VIRGINIA
#     WRITTEN ... JANUARY 1992
#------------------------
     #  DOUBLE PRECISION A, X, P, Q
     #  REAL P0, Q0, X0
     #  DOUBLE PRECISION AM1, APN, AP1, AP2, AP3, B, C, C1, C2, C3, C4,
     # *                 C5, D, E, EPS, G, H, LN10, PN, QG, QN, R, RTA,
     # *                 S, SUM, S2, T, TOL, U, W, XMIN, XN, Y, Z, AMIN
     #  DOUBLE PRECISION DPMPAR, DLNREL, DGAMMA, DGAMLN, DGMLN1, DRCOMP
#------------------------
#     LN10 = LN(10)
#     C = EULER CONSTANT
#------------------------
    ln10 = 2.302585
    c = γ
#------------------------
    tol = 1.e-10
#------------------------
#
#     ****** E AND XMIN ARE MACHINE DEPENDENT CONSTANTS. E IS THE
#            SMALLEST NUMBER FOR WHICH 1.0 + E .GT. 1.0, AND XMIN
#            IS THE SMALLEST POSITIVE NUMBER.
#
    e = eps(typeof(float(a)))
    xmin = realmin(float(a))
#
#------------------------
    x = 0.0
    if a <= 0.0 throw(DomainError()) end
    if p < 0.0 || q < 0.0 throw(DomainError()) end
    t = ((p + q) - 0.5) - 0.5
    if abs(t) > 5.0*max(e,1.e-30) throw(DomainError()) end
    ierr = 0
    xmin /= e
    if p/e <= xmin return 0.0 end
    if q/e <= xmin return realmax() end
    if a == 1.0
        if q >= 0.9 # go to 411
            return -dlnrel(-p)
        end
        return -log(q)
    end
    e = max(e, 1.e-30)
    deps = 1.e3*e
    amin = 5.e3
    if e < 1.e-17 amin = 2.e6 end
    xn = 0.0
    useP = true
    while true
        if a < amin # go to 50

#        GET AN INITIAL APPROXIMATION USING THE SINGLE
#         PRECISION ARITHMETIC (IF THIS IS POSSIBLE)

            p0 = float32(p)
            q0 = float32(q)
            if p0 != 0.0 && q0 != 0.0 # go to 10
                x0, ier = gaminv(float32(a), 0.0f0, p0, q0)
                if ier >= 0 || ier == -8 # go to 10
                    ierr = max(ier,0)
                    if x0 <= 1.f34 # go to 10
                        xn = float64(x0)
                        break
                    end
                end
            end

            if a <= 1.0 # go to 50
                xn = 0.0

#        SELECTION OF THE INITIAL APPROXIMATION XN OF X
#                       WHEN A .LT. 1
            
                g = gamma(a + 1.0)
                qg = q*g
                if qg == 0.0 return realmax() end
                b = qg/a
                if qg <= 0.6*a # go to 30
                    if a < 0.30 && b >= 0.35 # go to 20
                        t = exp(-(b + c))
                        u = t*exp(t)
                        xn = t*exp(u)
                        break
                    end
                end
                if b < 0.45 # go to 30
                    if b == 0.0 return realmax() end
                    y = -log(b)
                    s = 0.5 + (0.5 - a)
                    z = log(y)
                    t = y - s*z
                    if b >= 0.15 # go to 21
                        xn = y - s*log(t) - log(1.0 + s/(t + 1.0))
                        useP = false
                        break
                    end
                    if b > 1.e-2 # go to 22
                        u = ((t + 2.0*(3.0 - a))*t + (2.0 - a)*(3.0 - a))/((t + (5.0 - a))*t + 2.0)
                        xn = y - s*log(t) - log(u)
                        useP = false
                        break
                    end
                    c1 = -s*z
                    c2 = -s*(1.0 + c1)
                    c3 =  s*((0.5*c1 + (2.0 - a))*c1 + (2.5 - 1.5*a))
                    c4 = -s*(((c1/3.0 + (2.5 - 1.5*a))*c1 + ((a - 6.0)*a + 7.0))*c1 + ((11.0*a - 46.0)*a + 47.0)/6.0)
                    c5 = -s*((((-c1/4.0 + (11.0*a - 17.0)/6.0)*c1 + ((-3.0*a + 13.0)*a - 13.0))*c1 + 0.5*(((2.0*a - 25.0)*a + 72.0)*a - 61.0))*c1 + (((25.0*a - 195.0)*a + 477.0)*a - 379.0)/12.0)
                    xn = ((((c5/y + c4)/y + c3)/y + c2)/y + c1) + y
                    useP = false
                    break
                end
                if b*q <= 1.0e-8 # go to 31
                    xn = exp(-(q/a + c))
                elseif p > 0.9 # go to 32
                    xn = exp((dlnrel(-q) + lgamma1p(a))/a)
                else
                    xn = exp(log(p*g)/a)
                end
    
                if xn == 0.0 return 0.0 end # This one is in fact ierr=-3 in NSWC
                t = 0.5 + (0.5 - xn/(a + 1.0))
                xn /= t
                break
            end
        end

#        SELECTION OF THE INITIAL APPROXIMATION XN OF X
#                       WHEN A .GT. 1

        s = p <= 0.5 ? Φinv(p) : -Φinv(q)
        rta = sqrt(a)
        s2 = s*s
        xn = (((12.0*s2 - 243.0)*s2 - 923.0)*s2 + 1472.0)/204120.0 - s*(((3753.0*s2 + 4353.0)*s2 - 289517.0)*s2 - 289717.0)/(146966400.0*rta)
        xn = (xn/a + s*((9.0*s2 + 256.0)*s2 - 433.0)/(38880.0*rta)) - ((3.0*s2 + 7.0)*s2 - 16.0)/810.0
        xn = a + s*rta + (s2 - 1.0)/3.0 + s*(s2 - 7.0)/(36.0*rta) + xn/a
        xn = max(xn, 0.0)
        if a >= amin # go to 60
            x = xn
            d = 0.5 + (0.5 - x/a)
            if abs(d) <= 1.e-1 # go to 60
                if abs(d) > 1.e-3 break end
                return x
            end
        end

        if p > 0.5 # go to 70
            if xn < 3.*a
                useP = false
                break
            end
            w = log(q)
            y = -(w + lgamma(a))
            d = max(2.0, a*(a - 1.0))
            if y >= ln10*d # go to 61
                s = 1.0 - a
                z = log(y)
                c1 = -s*z
                c2 = -s*(1.0 + c1)
                c3 =  s*((0.5*c1 + (2.0 - a))*c1 + (2.5 - 1.5*a))
                c4 = -s*(((c1/3.0 + (2.5 - 1.5*a))*c1 + ((a - 6.0)*a + 7.0))*c1 + ((11.0*a - 46.0)*a + 47.0)/6.0)
                c5 = -s*((((-c1/4.0 + (11.0*a - 17.0)/6.0)*c1 + ((-3.0*a + 13.0)*a - 13.0))*c1 + 0.5*(((2.0*a - 25.0)*a + 72.0)*a - 61.0))*c1 + (((25.0*a - 195.0)*a + 477.0)*a - 379.0)/12.0)
                xn = ((((c5/y + c4)/y + c3)/y + c2)/y + c1) + y
                useP = false
                break
            end
            t = a - 1.0
            xn = y + t*log(xn) - dlnrel(-t/(xn + 1.0))
            xn = y + t*log(xn) - dlnrel(-t/(xn + 1.0))
            useP = false
            break
        end

        ap1 = a + 1.0
        if xn > 0.7*ap1 break end
        w = log(p) + lgamma(ap1)
        if xn <= 0.15*ap1 # go to 80
            ap2 = a + 2.0
            ap3 = a + 3.0
            x = exp((w + x)/a)
            x = exp((w + x - log(1.0 + (x/ap1)*(1.0 + x/ap2)))/a)
            x = exp((w + x - log(1.0 + (x/ap1)*(1.0 + x/ap2)))/a)
            x = exp((w + x - log(1.0 + (x/ap1)*(1.0 + (x/ap2)*(1.0 + x/ap3))))/a)
            xn = x
            if xn <= 1.e-2*ap1 break end
        end

        apn = ap1
        t = xn/apn
        sum = 1.0 + t
        while true
            apn += 1.0
            t *= xn/apn
            sum += t
            if t <= 1.e-4 break end # go to 81
        end
        t = w - log(sum)
        xn = exp((xn + t)/a)
        xn *= 1.0 - (a*log(xn) - xn - t)/(a - xn)
        break
    end

#                 SCHRODER ITERATION USING P
    if p > 0.5 useP = false end
    if useP
        if p <= xmin return xn end # go to 550
        am1 = (a - 0.5) - 0.5

        while true
            if ierr >= 10 return x end # go to 530
            ierr += 1
            pn, qn = dgrat(a, xn)
            if pn == 0.0 || qn == 0.0 return xn end# go to 550
            r = drcomp(a,xn)
            if r < xmin return xn end # go to 550
            t = (pn - p)/r
            w = 0.5*(am1 - xn)
            if abs(t) > 0.1 || abs(w*t) > 0.1 # go to 120
                x = xn*(1.0 - t)
                if x <= 0.0 error("Iteration failed") end # go to 540
                d = abs(t)
            else # go to 121
                h = t*(1.0 + w*t)
                x = xn*(1.0 - h)
                if x <= 0.0 error("Iteration failed") end # go to 540
                if abs(w) >= 1.0 && abs(w)*t*t <= deps return x end
                d = abs(h)
            end
            xn = x
            if d <= tol # go to 110
                if d <= deps return x end
                if abs(p - pn) <= tol*p return x end
            end
        end
    else

#                 SCHRODER ITERATION USING Q

        if q <= xmin return xn end # go to 550
        am1 = (a - 0.50) - 0.50

        while true
            if ierr >= 10 return x end # go to 530
            ierr += 1
            pn, qn = dgrat(a, xn)
            if pn == 0.0 || qn == 0.0 return xn end # go to 550
            r = drcomp(a,xn)
            if r < xmin return xn end # go to 550
            t = (q - qn)/r
            w = 0.5*(am1 - xn)
            if abs(t) > 0.1 || abs(w*t) > 0.1 # go to 220
                x = xn*(1.0 - t)
                if x <= 0.0 error("Iteration failed") end # go to 540
                d = abs(t)
            else
                h = t*(1.0 + w*t)
                x = xn*(1.0 - h)
                if x <= 0.0 error("Iteration failed") end # go to 540
                if abs(w) >= 1.0 && abs(w)*t*t <= deps return x end
                d = abs(h)
            end
            xn = x
            if d <= tol # go to 210
                if d <= deps return x end
                if abs(q - qn) <= tol*q return x end
            end
        end
    end
end

function gaminv(a::Float32, x0::Float32, p::Float32, q::Float32)
#-----------------------------------------------------------------------
#
#             INVERSE INCOMPLETE GAMMA RATIO FUNCTION
#
#     GIVEN POSITIVE A, AND NONEGATIVE P AND Q WHERE P + Q = 1.
#     THEN X IS COMPUTED WHERE P(A,X) = P AND Q(A,X) = Q. SCHRODER
#     ITERATION IS EMPLOYED. THE ROUTINE ATTEMPTS TO COMPUTE X
#     TO 10 SIGNIFICANT DIGITS IF THIS IS POSSIBLE FOR THE
#     PARTICULAR COMPUTER ARITHMETIC BEING USED.
#
#                        ------------
#
#     X IS A VARIABLE. IF P = 0 THEN X IS ASSIGNED THE VALUE 0,
#     AND IF Q = 0 THEN X IS SET TO THE LARGEST FLOATING POINT
#     NUMBER AVAILABLE. OTHERWISE, GAMINV ATTEMPTS TO OBTAIN
#     A SOLUTION FOR P(A,X) = P AND Q(A,X) = Q. IF THE ROUTINE
#     IS SUCCESSFUL THEN THE SOLUTION IS STORED IN X.
#
#     X0 IS AN OPTIONAL INITIAL APPROXIMATION FOR X. IF THE USER
#     DOES NOT WISH TO SUPPLY AN INITIAL APPROXIMATION, THEN SET
#     X0 .LE. 0.
#
#     IERR IS A VARIABLE THAT REPORTS THE STATUS OF THE RESULTS.
#     WHEN THE ROUTINE TERMINATES, IERR HAS ONE OF THE FOLLOWING
#     VALUES ...
#
#       IERR =  0    THE SOLUTION WAS OBTAINED. ITERATION WAS
#                    NOT USED.
#       IERR.GT.0    THE SOLUTION WAS OBTAINED. IERR ITERATIONS
#                    WERE PERFORMED.
#       IERR = -2    (INPUT ERROR) A .LE. 0
#       IERR = -3    NO SOLUTION WAS OBTAINED. THE RATIO Q/A
#                    IS TOO LARGE.
#       IERR = -4    (INPUT ERROR) P OR Q IS NEGATIVE, OR
#                    P + Q .NE. 1.
#       IERR = -6    20 ITERATIONS WERE PERFORMED. THE MOST
#                    RECENT VALUE OBTAINED FOR X IS GIVEN.
#                    THIS CANNOT OCCUR IF X0 .LE. 0.
#       IERR = -7    ITERATION FAILED. NO VALUE IS GIVEN FOR X.
#                    THIS MAY OCCUR WHEN X IS APPROXIMATELY 0.
#       IERR = -8    A VALUE FOR X HAS BEEN OBTAINED, BUT THE
#                    ROUTINE IS NOT CERTAIN OF ITS ACCURACY.
#                    ITERATION CANNOT BE PERFORMED IN THIS
#                    CASE. IF X0 .LE. 0, THIS CAN OCCUR ONLY
#                    WHEN P OR Q IS APPROXIMATELY 0. IF X0 IS
#                    POSITIVE THEN THIS CAN OCCUR WHEN A IS
#                    EXCEEDINGLY CLOSE TO X AND A IS EXTREMELY
#                    LARGE (SAY A .GE. 1.E20).
#
#-----------------------------------------------------------------------
#     WRITTEN BY ALFRED H. MORRIS, JR.
#        NAVAL SURFACE WARFARE CENTER
#        DAHLGREN, VIRGINIA
#     REVISED ... JANUARY 1992
#------------------------
      # REAL LN10, BMIN(2), EMIN(2)
#------------------------
#     LN10 = LN(10)
#     C = EULER CONSTANT
#------------------------
    ln10 = 2.302585f0
    c = .577215664901533f0
#------------------------
    bmin = [1.f-28, 1.f-13]
    emin = [2.f-03, 6.f-03]
#------------------------
    tol  = 1.f-5
#------------------------
#
#     ****** E AND XMIN ARE MACHINE DEPENDENT CONSTANTS. E IS THE
#            SMALLEST NUMBER FOR WHICH 1.0 + E .GT. 1.0, AND XMIN
#            IS THE SMALLEST POSITIVE NUMBER.
#
    e = eps(Float32)
    xmin = realmin(Float32)

#------------------------
    x = 0.0f0
    if a <= 0.0 throw(DomainError()) end # go to 500
    if p < 0.0 || q < 0.0f0 throw(DomainError()) end # go to 520
    t = ((p + q) - 0.5f0) - 0.5f0
    if abs(t) > 5.0f0*max(e,1.f-15) throw(DomainError()) end # go to 520

    ierr = 0
    xmin /= e
    if p/e <= xmin return x, ierr end # go to 400
    if q/e <= xmin return realmax(Float32), -8 end # go to 560
    if a == 1.0f0 # go to 410
        if q >= 0.9f0 # go to 411
            return -alnrel(-p), ierr
        end
        return -log(q), ierr
    end

    e2 = e + e
    amax = 0.4f-10/(e*e)
    deps = max(100.0f0*e,1.f-10)
    iop = 1
    if e > 1.f-10 iop = 2 end
    xn = x0
    useP = true
    while true
        if x0 > 0.0 break end # go to 100

#        SELECTION OF THE INITIAL APPROXIMATION XN OF X
#                       WHEN A .LT. 1

        if a <= 1.0 # go to 50
            g = gamma(a + 1.0f0)
            qg = q*g
            if qg == 0.0 return realmax(Float32), -8 end # go to 560
            b = qg/a
            if qg <= 0.6f0*a # go to 20
                if a < 0.30 && b >= 0.35 # go to 10
                    t = exp(-(b + c))
                    u = t*exp(t)
                    xn = t*exp(u)
                    break
                end
                if b < 0.45 # go to 20
                    if b == 0.0 return realmax(Float32), -8 end # go to 560
                    y = -log(b)
                    s = 0.5f0 + (0.5f0 - a)
                    z = log(y)
                    t = y - s*z
                    if b >= 0.15 # go to 11
                        xn = y - s*log(t) - log(1.0f0 + s/(t + 1.0f0))
                        useP = false
                        break
                    end
                    if b > 1.f-2 # go to 12
                        u = ((t + 2.0f0*(3.0f0 - a))*t + (2.0f0 - a)*(3.0f0 - a))/((t + (5.0f0 - a))*t + 2.0f0)
                        xn = y - s*log(t) - log(u)
                        useP = false
                        break
                    end
                    c1 = -s*z
                    c2 = -s*(1.0f0 + c1)
                    c3 =  s*((0.5f0*c1 + (2.0f0 - a))*c1 + (2.5f0 - 1.5f0*a))
                    c4 = -s*(((c1/3.0f0 + (2.5f0 - 1.5f0*a))*c1 + ((a - 6.0f0)*a + 7.0f0))*c1 + ((11.0f0*a - 46.0f0)*a + 47.0f0)/6.0f0)
                    c5 = -s*((((-c1/4.0f0 + (11.0f0*a - 17.0f0)/6.0f0)*c1 + ((-3.0f0*a + 13.0f0)*a - 13.0f0))*c1 + 0.5f0*(((2.0f0*a - 25.0f0)*a + 72.0f0)*a - 61.0f0))*c1 + (((25.0f0*a - 195.0f0)*a + 477.0f0)*a - 379.0f0)/12.0f0)
                    xn = ((((c5/y + c4)/y + c3)/y + c2)/y + c1) + y
                    if a > 1.0 || b > bmin[iop]
                        useP = false
                        break
                    end
                    return xn, ierr
                end
            end
            
            if b*q <= 1.f-8 # go to 21
                xn = exp(-(q/a + c))
            elseif p > 0.9 # go to 22
                xn = exp((alnrel(-q) + gamln1(a))/a)
            else
                xn = exp(log(p*g)/a)
            end
            if xn == 0.0 return 0.0f0, -3 end
            t = 0.5f0 + (0.5f0 - xn/(a + 1.0f0))
            xn /= t
            break
        end

#        SELECTION OF THE INITIAL APPROXIMATION XN OF X
#                       WHEN A .GT. 1

        t = p - 0.5f0
        if q < 0.5 t = 0.5f0 - q end
        s = p <= 0.5 ? Φinv(p) : -Φinv(q)

        rta = sqrt(a)
        s2 = s*s
        xn = (((12.0f0*s2 - 243.0f0)*s2 - 923.0f0)*s2 + 1472.0f0)/204120.0f0
        xn = (xn/a + s*((9.0f0*s2 + 256.0f0)*s2 - 433.0f0)/(38880.0f0*rta)) - ((3.0f0*s2 + 7.0f0)*s2 - 16.0f0)/810.0f0
        xn = a + s*rta + (s2 - 1.0f0)/3.0f0 + s*(s2 - 7.0f0)/(36.0f0*rta) + xn/a
        xn = max(xn, 0.0f0)

        amin = 20.0f0
        if e < 1.f-8 amin = 250.0f0 end
        if a >= amin # go to 60
            x = xn
            d = 0.5f0 + (0.5f0 - x/a)
            if abs(d) <= 1.f-1 return x, ierr end
        end

        if p > 0.5 # go to 70
            if xn < 3.0f0*a
                useP = false
                break
            end
            w = log(q)
            y = -(w + lgamma(a))
            d = max(2.0f0, a*(a - 1.0f0))
            if y >= ln10*d # go to 61
                s = 1.0f0 - a
                z = log(y)
                c1 = -s*z
                c2 = -s*(1.0f0 + c1)
                c3 =  s*((0.5f0*c1 + (2.0f0 - a))*c1 + (2.5f0 - 1.5f0*a))
                c4 = -s*(((c1/3.0f0 + (2.5f0 - 1.5f0*a))*c1 + ((a - 6.0f0)*a + 7.0f0))*c1 + ((11.0f0*a - 46.0f0)*a + 47.0f0)/6.0f0)
                c5 = -s*((((-c1/4.0f0 + (11.0f0*a - 17.0f0)/6.0f0)*c1 + ((-3.0f0*a + 13.0f0)*a - 13.0f0))*0.5f0*(((2.0f0*a - 25.0f0)*a + 72.0f0)*a - 61.0f0))*c1 + (((25.0f0*a - 195.0f0)*a +.0f0)*a - 379.0f0)/12.0f0)
                xn = ((((c5/y + c4)/y + c3)/y + c2)/y + c1) + y
                if a > 1.0 || b > bmin[iop]
                    useP = false
                    break
                end
                return xn, ierr
            end
            t = a - 1.0f0
            xn = y + t*log(xn) - alnrel(-t/(xn + 1.0f0))
            xn = y + t*log(xn) - alnrel(-t/(xn + 1.0f0))
            useP = false
            break
        end

        ap1 = a + 1.0f0
        if xn > 0.70f0*ap1 break end # go to 101
        w = log(p) + lgamma(ap1)
        if xn <= 0.15f0*ap1 # go to 80
            ap2 = a + 2.0f0
            ap3 = a + 3.0f0
            x = exp((w + x)/a)
            x = exp((w + x - log(1.0f0 + (x/ap1)*(1.0f0 + x/ap2)))/a)
            x = exp((w + x - log(1.0f0 + (x/ap1)*(1.0f0 + x/ap2)))/a)
            x = exp((w + x - log(1.0f0 + (x/ap1)*(1.0f0 + (x/ap2)*(1.0f0 + x/ap3))))/a)
            xn = x
            if xn <= 1.f-2*ap1 # go to 80
                if xn <= emin[iop]*ap1 return x, ierr end
                break
            end
        end

        apn = ap1
        t = xn/apn
        sum = 1.0f0 + t
        while true
            apn += 1.0f0
            t *= xn/apn
            sum += t
            if t <= 1.f-4 break end
        end
        t = w - log(sum)
        xn = exp((xn + t)/a)
        xn *= 1.0f0 - (a*log(xn) - xn - t)/(a - xn)
        break
    end

#                 SCHRODER ITERATION USING P

    if p > 0.5 useP = false end # go to 200
    if useP
        while true
            if p <= xmin return xn, -8 end # go to 550
            am1 = (a - 0.5f0) - 0.5f0
            if a > amax # go to 110
                d = 0.5f0 + (0.5f0 - xn/a)
                if abs(d) <= e2 return xn, -8 end # go to 550
            end
            if ierr >= 20 return xn, -6 end # go to 530
            ierr += 1
            pn, qn = gratio(a, xn, 0)
            if pn == 0.0 || qn == 0.0 return xn, -8 end # go to 550
            r = rcomp(a, xn)
            if r < xmin return xn, -8 end # go to 550
            t = (pn - p)/r
            w = 0.5f0*(am1 - xn)
            if abs(t) > 0.1 || abs(w*t) > 0.1 # go to 120
                x = xn*(1.0f0 - t)
                if x <= 0.0 error("Iteration failed") end # go to 540
                d = abs(t)
            else
                h = t*(1.0f0 + w*t)
                x = xn*(1.0f0 - h)
                if x <= 0.0 error("Iteration failed") end # go to 540
                if abs(w) >= 1.0 && abs(w)*t*t <= deps return x, ierr end
                d = abs(h)
            end
            xn = x
            if d <= tol # go to 102
                if d <= deps || abs(p - pn) < tol*p return x, ierr end
            end
        end
    end

#                 SCHRODER ITERATION USING Q

    while true
        if q <= xmin return xn, -8 end # go to 550
        am1 = (a - 0.5f0) - 0.5f0
        if a > amax # go to 210
            d = 0.5f0 + (0.5f0 - xn/a)
            if abs(d) <= e2 return xn, -8 end # go to 550
        end

        if ierr >= 20 return x, -6 end # go to 530
        ierr += 1
        pn, qn = gratio(a, xn, 0)
        if pn == 0.0 || qn == 0.0 return xn, -8 end # go to 550
        r = rcomp(a, xn)
        if r < xmin return xn, -8 end # go to 550
        t = (q - qn)/r
        w = 0.5f0*(am1 - xn)
        if abs(t) > 0.1 || abs(w*t) > 0.1 # go to 220
            x = xn*(1.0f0 - t)
            if x <= 0.0 error("Iteration failed") end # go to 540
            d = abs(t)
        else
            h = t*(1.0f0 + w*t)
            x = xn*(1.0f0 - h)
            if x <= 0.0 error("Iteration failed") end # go to 540
            if abs(w) >= 1.0 && abs(w)*t*t <= deps return x, ierr end
            d = abs(h)
        end
        xn = x
        if d <= tol # go to 201
            if d <= deps || abs(q - qn) <= tol*q return x, ierr end
        end
    end
end


# The regularized incomplete beta function
# Translated from the NSWC Library
function bratio(a::Real, b::Real, x::Real)
#     SUBROUTINE BRATIO (A, B, X, Y, W, W1, IERR) 
#-----------------------------------------------------------------------
#
#            EVALUATION OF THE INCOMPLETE BETA FUNCTION IX(A,B)
#                     --------------------
#
#     IT IS ASSUMED THAT A AND B ARE NONNEGATIVE, AND THAT X <= 1
#     AND Y = 1 - X.  BRATIO ASSIGNS W AND W1 THE VALUES
#
#                      W  = IX(A,B)
#                      W1 = 1 - IX(A,B) 
#
#     IERR IS A VARIABLE THAT REPORTS THE STATUS OF THE RESULTS.
#     IF NO INPUT ERRORS ARE DETECTED THEN IERR IS SET TO 0 AND
#     W AND W1 ARE COMPUTED. OTHERWISE, IF AN ERROR IS DETECTED,
#     THEN W AND W1 ARE ASSIGNED THE VALUE 0 AND IERR IS SET TO
#     ONE OF THE FOLLOWING VALUES ...
#
#        IERR = 1  IF A OR B IS NEGATIVE
#        IERR = 2  IF A = B = 0
#        IERR = 3  IF X < 0 OR X > 1
#        IERR = 4  IF Y < 0 OR Y > 1
#        IERR = 5  IF X + Y != 1
#        IERR = 6  IF X = A = 0
#        IERR = 7  IF Y = B = 0
#
#--------------------
#     WRITTEN BY ALFRED H. MORRIS, JR.
#        NAVAL SURFACE WARFARE CENTER
#        DAHLGREN, VIRGINIA
#     REVISED ... NOV 1991
#-----------------------------------------------------------------------
#     REAL LAMBDA
#-----------------------------------------------------------------------
#
#     ****** EPS IS A MACHINE DEPENDENT CONSTANT. EPS IS THE SMALLEST 
#            FLOATING POINT NUMBER FOR WHICH 1.0 + EPS > 1.0
#
#                      EPS = SPMPAR(1)
#
#-----------------------------------------------------------------------
    precision = eps(float(x))
    w = 0.0
    w1 = 0.0
    y = 1.0 - x
    if a < 0.0 || b < 0.0 error("a and b must be non-negative") end
    if a == 0.0 && b == 0.0 error("Either a or b must positive") end
    if x < 0.0 || x > 1.0 error("x must be between zero and one") end

    if x == 0.0
        if a == 0.0 error("Either x or a must be positive") end
        return 0.0
    end
    if y == 0.0
        if a == 0.0 error("Either x must be less than one or b must be positive") end
        return 1.0
    end
    if a == 0.0 return 1.0 end
    if b == 0.0 return 0.0 end
    
    if max(a,b) < 1.e-3*precision return b/(a + b) end

    ind = false
    a0 = a
    b0 = b
    x0 = x
    y0 = y
    if min(a0, b0) <= 1.0

#             PROCEDURE FOR A0 <= 1 OR B0 <= 1

        if x > 0.5
            ind = true
            a0 = b
            b0 = a
            x0 = y
            y0 = x
        end

        if b0 < min(precision,precision*a0) # go to 80
            w = fpser(a0, b0, x0, precision)
            return ind ? 1.0 - w : w
        end
        if a0 < min(precision,precision*b0) && b0*x0 <= 1.0 # go to 90
            w1 = apser(a0, b0, x0, precision)
            return ind ? w1 : 1.0 - w1
        end
        if max(a0, b0) <= 1.0 # go to 20
            if a0 >= min(0.2, b0) # go to 100
                w = bpser(a0, b0, x0, precision)
                return ind ? 1.0 - w : w
            end
            if x0^a0 <= 0.9 # go to 100
                w = bpser(a0, b0, x0, precision)
                return ind ? 1.0 - w : w
            end
            if x0 >= 0.3 # go to 110
                w1 = bpser(b0, a0, y0, precision)
                return ind ? w1 : 1.0 - w1
            end
            n = 20
            w1 = bup(b0, a0, y0, x0, n, precision) # go to 130
            b0 += n
            w1 = bgrat(b0, a0, y0, x0, w1, 15.0*precision)
            return ind ? w1 : 1.0 - w1
        end 
    
        if b0 <= 1.0 # go to 100
            w = bpser(a0, b0, x0, precision)
            return ind ? 1.0 - w : w
        end
        if x0 >= 0.3 # go to 110
            w1 = bpser(b0, a0, y0, precision)
            return ind ? w1 : 1.0 - w1
        end
        if x0 < 0.1 # go to 21
            if (x0*b0)^a0 <= 0.7 # go to 100
                w = bpser(a0, b0, x0, precision)
                return ind ? 1.0 - w : w
            end
        end
        if b0 > 15.0 # go to 131
            w1 = bgrat(b0, a0, y0, x0, w1, 15.0*precision)
            return ind ? w1 : 1.0 - w1
        end
        n = 20
        w1 = bup(b0, a0, y0, x0, n, precision) #  go to 130
        b0 += n
        w1 = bgrat(b0, a0, y0, x0, w1, 15.0*precision)
        return ind ? w1 : 1.0 - w1
    end

#             PROCEDURE FOR A0 > 1 AND B0 > 1

    if a <= b # go to 31
        lambda = a - (a + b)*x
    else
        lambda = (a + b)*y - b
    end
    if lambda < 0.0 # go to 40
        ind = true
        a0 = b
        b0 = a
        x0 = y
        y0 = x
        lambda = abs(lambda)
    end

    if b0 < 40.0 && b0*x0 <= 0.7 # go to 100
        w = bpser(a0, b0, x0, precision)
        return ind ? 1.0 - w : w
    end
    if b0 < 40.0 # go to 140
        n = itrunc(b0)
        b0 -= n
        if b0 == 0.0 # go to 141
            n -= 1
            b0 = 1.0
        end
        w = bup(b0, a0, y0, x0, n, precision)
        if x0 <= 0.7 # go to 150
            w += bpser(a0, b0, x0, precision)
            return ind ? 1.0 - w : w
        end
        if a0 <= 15.0 # go to 151
            n = 20
            w += bup(a0, b0, x0, y0, n, precision)
            a0 += n
        end
        w = bgrat(a0, b0, x0, y0, w, 15.0*precision)
        return ind ? 1.0 - w : w
    end
    if a0 <= b0 # go to 50
        if a0 <= 100.0 || lambda > 0.03*a0 # go to 120
            w = bfrac(a0, b0, x0, y0, lambda, 15.0*precision)
            return ind ? 1.0 - w : w
        end
        w = basym(a0, b0, lambda, 100.0*precision)
        return ind ? 1.0 - w : w
    end
    if b0 <= 100.0 # go to 120
        w = bfrac(a0, b0, x0, y0, lambda, 15.0*precision)
        return ind ? 1.0 - w : w
    end
    if lambda > 0.03*b0 # go to 120
        w = bfrac(a0, b0, x0, y0, lambda, 15.0*precision)
        return ind ? 1.0 - w : w
    end
    w = basym(a0, b0, lambda, 100.0*precision)
    return ind ? 1.0 - w : w
end

## Auxilliary functions

function fpser(a::Real, b::Real, x::Real, precision::Real)
#     REAL FUNCTION FPSER (A, B, X, EPS)
#-----------------------------------------------------------------------
#
#                 EVALUATION OF I (A,B) 
#                                X
#
#          FOR B .LT. MIN(EPS,EPS*A) AND X .LE. 0.5.
#
#-----------------------------------------------------------------------
#
#                  SET  FPSER = X**A
#
    fpserval = x^a
    if fpserval < precision return 0.0 end
#                note that 1/b(a,b) = b 

    fpserval *= b/a
    tol = precision/a
    an = a + 1.0
    t = x
    s = t/an
    while abs(c) > tol
        an += 1.0
        t *= x
        c = t/an
        s += c
    end
    fpserval *= 1.0 + a*s
    return fpserval
end

function apser(a::Real, b::Real, x::Real, precision::Real)
#     REAL FUNCTION APSER (A, B, X, EPS)
#-----------------------------------------------------------------------
#     APSER YIELDS THE INCOMPLETE BETA RATIO I(SUB(1-X))(B,A) FOR
#     A .LE. MIN(EPS,EPS*B), B*X .LE. 1, AND X .LE. 0.5. USED WHEN
#     A IS VERY SMALL. USE ONLY IF ABOVE INEQUALITIES ARE SATISFIED.
#-----------------------------------------------------------------------
#     REAL J
#--------------------
    g = 0.57721566490153286
#--------------------
    bx = b*x
    t = x - bx
    if (b*precision <= 2.e-2) # go to 10
        c = log(x) + digamma(b) + g + t
    else
        c = log(bx) + g + t
    end

    tol = 5.0*precision*abs(c)
    j = 1.0
    s = 0.0
    aj = t
    while (abs(aj) > tol)
        j += 1.0
        t *= x - bx/j
        aj = t/j
        s += aj 
    end
    return -a*(c + s)
end

function bpser(a::Real, b::Real, x::Real, precision::Real)
#     REAL FUNCTION BPSER(A, B, X, EPS) 
#-----------------------------------------------------------------------
#     POWER SERIES EXPANSION FOR EVALUATING IX(A,B) WHEN B .LE. 1
#     OR B*X .LE. 0.7.  EPS IS THE TOLERANCE USED.
#-----------------------------------------------------------------------
#     REAL N
#
#     BPSER = 0.0
    if (x == 0.0) return 0.0 end
#-----------------------------------------------------------------------
#            COMPUTE THE FACTOR X**A/(A*BETA(A,B))
#-----------------------------------------------------------------------
    a0 = min(a,b)
    if (a0 >= 1.0) # go to 10
         z = a*log(x) - dbetln(a,b)
         bpserval = exp(z)/a
         # go to 70
    else
        b0 = max(a,b)
        if (b0 <= 1.0)

#            PROCEDURE FOR A0 .LT. 1 AND B0 .LE. 1

            bpserval = x^a
            if (bpserval == 0.0) return 0.0 end

            apb = a + b
            if (apb <= 1.0) # go to 20
                z = 1.0 + rgamma1pm1(apb)
            else
                u = a + b - 1.0
                z = (1.0 + rgamma1pm1(u))/apb 
            end
            c = (1.0 + rgamma1pm1(a))*(1.0 + rgamma1pm1(b))/z
            bpserval *= c*(b/apb) 

        elseif (b0 < 8.0)
#         PROCEDURE FOR A0 .LT. 1 AND 1 .LT. B0 .LT. 8

            u = lgamma1p(a0)
            m = itrunc(b0 - 1.0)
            if (m >= 1) # go to 50
                c = 1.0
                for i = 1:m 
                    b0 -= 1.0
                    c *= b0/(a0 + b0)
                end
                u += log(c)
            end
            z = a*log(x) - u
            b0 -= 1.0 
            apb = a0 + b0 
            if (apb <= 1.0) # go to 51
                t = 1.0 + rgamma1pm1(apb)
            else
                u = a0 + b0 - 1.0
                t = (1.0 + rgamma1pm1(u))/apb
            end
            bpserval = exp(z)*(a0/a)*(1.0 + rgamma1pm1(b0))/t
        else

#            PROCEDURE FOR A0 .LT. 1 AND B0 .GE. 8

            u = lgamma1p(a0) + dlgdiv(a0,b0)
            z = a*log(x) - u
            bpserval = (a0/a)*exp(z)
        end
    end
    if (bpserval == 0.0 || a <= 0.1*precision) return bpserval end
#-----------------------------------------------------------------------
#                     COMPUTE THE SERIES
#-----------------------------------------------------------------------
    sumval = 0.0
    n = 0.0
    c = 1.0
    tol = precision/a
    w = c
    while abs(w) > tol
        n += 1.0
        c *= (0.5 + (0.5 - b/n))*x
        w = c/(a + n)
        sumval += w
    end
    return bpserval*(1.0 + a*sumval)
end

function bup(a::Real, b::Real, x::Real, y::Real, n::Integer, precision::Real)
#     REAL FUNCTION BUP(A, B, X, Y, N, EPS)
#-----------------------------------------------------------------------
#     EVALUATION OF IX(A,B) - IX(A+N,B) WHERE N IS A POSITIVE INTEGER.
#     EPS IS THE TOLERANCE USED.
#-----------------------------------------------------------------------
#     REAL L
#
#          OBTAIN THE SCALING FACTOR EXP(-MU) AND 
#             EXP(MU)*(X**A*Y**B/BETA(A,B))/A
#
    apb = a + b
    ap1 = a + 1.0
    mu = 0
    d = 1.0
    if (n != 1 && a >= 1.0 && apb >= 1.1*ap1) # go to 10
        mu = itrunc(abs(realminexp(Float64)))
        k = itrunc(realmaxexp(Float64))
        if (k < mu) mu = k end
        t = mu
        d = exp(-t)
    end
    bupval = brcmp1(mu,a,b,x,y)/a
    if (n == 1 || bupval == 0.0) return bupval end
    nm1 = n - 1
    w = d

#          LET K BE THE INDEX OF THE MAXIMUM TERM 

    k = 0
    while true
        if (b <= 1.0) break end# go to 40
        if (y <= 1.e-4) # go to 20
            k = nm1
        else
            r = (b - 1.0)*x/y - a
            if (r < 1.0) break end
            k = nm1
            t = nm1
            if (r < t) k = itrunc(r) end
        end

#          ADD THE INCREASING TERMS OF THE SERIES 

        for i = 1:k
            l = i - 1
            d *= ((apb + l)/(ap1 + l))*x
            w += d
        end
        if (k == nm1) return bupval*w end
        break
    end

#          ADD THE REMAINING TERMS OF THE SERIES
    kp1 = k + 1
    for i = kp1:nm1
        l = i - 1
        d *= ((apb + l)/(ap1 + l))*x
        w += d
        if (d <= precision*w) break end
    end

#               terminate the procedure 
    return bupval*w
end

function bfrac(a::Real, b::Real, x::Real, y::Real, lambda::Real, precision::Real)
#     REAL FUNCTION BFRAC(A, B, X, Y, LAMBDA, EPS)
#-----------------------------------------------------------------------
#     CONTINUED FRACTION EXPANSION FOR IX(A,B) WHEN A,B .GT. 1.
#     IT IS ASSUMED THAT  LAMBDA = (A + B)*Y - B. 
#-----------------------------------------------------------------------
#     REAL LAMBDA, N
#--------------------
    bfracval = brcomp(a,b,x,y) 
    if (bfracval == 0.0) return 0.0 end

    c = 1.0 + lambda
    c0 = b/a
    c1 = 1.0 + 1.0/a
    yp1 = y + 1.0

    n = 0.0
    p = 1.0
    s = a + 1.0
    an = 0.0
    bn = 1.0
    anp1 = 1.0
    bnp1 = c/c1
    r = c1/c

#        CONTINUED FRACTION CALCULATION 

    while true
        n += 1.0
        t = n/a
        w = n*(b - n)*x
        e = a/s
        alpha = (p*(p + c0)*e*e)*(w*x) 
        e = (1.0 + t)/(c1 + t + t)
        beta = n + w/s + e*(c + n*yp1) 
        p = 1.0 + t
        s += 2.0

#       update an, bn, anp1, and bnp1

        t = alpha*an + beta*anp1
        an = anp1
        anp1 = t
        t = alpha*bn + beta*bnp1
        bn = bnp1
        bnp1 = t

        r0 = r
        r = anp1/bnp1
        if (abs(r - r0) < precision*r) break end

#       rescale an, bn, anp1, and bnp1 

        an /= bnp1
        bn /= bnp1
        anp1 = r
        bnp1 = 1.0 
    end

#                 TERMINATION 

    return bfracval*r
end

function brcomp(a::Real, b::Real, x::Real, y::Real)
#     REAL FUNCTION BRCOMP (A, B, X, Y) 
#-----------------------------------------------------------------------
#               EVALUATION OF X**A*Y**B/BETA(A,B) 
#-----------------------------------------------------------------------
#     REAL LAMBDA, LNX, LNY
#-----------------
#     CONST = 1/SQRT(2*PI)
#-----------------
    cnst = .398942280401433

    brcompval = 0.0
    if (x == 0.0 || y == 0.0) return brcompval end
    a0 = min(a,b)
    if (a0 < 8.0) # go to 100

        if (x <= 0.375)# go to 10
            lnx = log(x)
            lny = dlnrel(-x)
        elseif (y <= 0.375)# go to 20
            lnx = dlnrel(-y)
            lny = log(y)
        else
            lnx = log(x)
            lny = log(y)
        end

        z = a*lnx + b*lny
        if (a0 >= 1.0) # go to 30
            z -= dbetln(a,b)
            return exp(z)
        end
            
#-----------------------------------------------------------------------
#              PROCEDURE FOR A .LT. 1 OR B .LT. 1 
#-----------------------------------------------------------------------
        b0 = max(a,b)
        #if (b0 .ge. 8.0) go to 80
        if (b0 <= 1.0) # go to 60

#                   algorithm for b0 .le. 1

            brcompval = exp(z)
            if (brcompval == 0.0) return 0.0 end

            apb = a + b
            if (apb <= 1.0) # go to 40
                z = 1.0 + rgamma1pm1(apb)
            else
                u = a + b - 1.0
                z = (1.0 + rgamma1pm1(u))/apb 
            end

            c = (1.0 + rgamma1pm1(a))*(1.0 + rgamma1pm1(b))/z
            return brcompval*(a0*c)/(1.0 + a0/b0)
        elseif (b0 < 8.0)

#                ALGORITHM FOR 1 .LT. B0 .LT. 8

            u = lgamma1p(a0)
            n = itrunc(b0 - 1.0)
            if (n > 1) # go to 70
                c = 1.0
                for i = 1:n 
                    b0 -= 1.0
                    c *= b0/(a0 + b0)
                end
                u += log(c)
            end

            z -= u
            b0 -= 1.0 
            apb = a0 + b0 
            if (apb <= 1.0)
                t = 1.0 + rgamma1pm1(apb)
            else
                u = a0 + b0 - 1.0
                t = (1.0 + rgamma1pm1(u))/apb
            end
            return a0*exp(z)*(1.0 + rgamma1pm1(b0))/t
        else

#                   ALGORITHM FOR B0 .GE. 8
            u = lgamma1p(a0) + dlgdiv(a0,b0)
            return a0*exp(z - u)
        end
    end
#-----------------------------------------------------------------------
#              PROCEDURE FOR A .GE. 8 AND B .GE. 8
#-----------------------------------------------------------------------
    if (a <= b) # go to 101 
        h = a/b
        x0 = h/(1.0 + h)
        y0 = 1.0/(1.0 + h)
        lambda = a - (a + b)*x
    else
        h = b/a
        x0 = 1.0/(1.0 + h)
        y0 = h/(1.0 + h)
        lambda = (a + b)*y - b
    end
    e = -lambda/a 
    if (abs(e) <= 0.6) # go to 111
        u = -log1pmx(e)
    else
        u = e - log(x/x0)
    end

    e = lambda/b
    if (abs(e) <= 0.6) # go to 121
        v = -log1pmx(e)
    else
        v = e - log(y/y0)
    end
    z = exp(-(a*u + b*v))
    return cnst*sqrt(b*x0)*z*exp(-dbcorr(a,b))
end

function brcmp1(mu::Integer, a::Real, b::Real, x::Real, y::Real)
#     REAL FUNCTION BRCMP1 (MU, A, B, X, Y)
#-----------------------------------------------------------------------
#          EVALUATION OF  EXP(MU) * (X**A*Y**B/BETA(A,B))
#-----------------------------------------------------------------------
#     REAL LAMBDA, LNX, LNY
#-----------------
#     CONST = 1/SQRT(2*PI)
#-----------------
    cnst = 0.3989422804014327

    brcompval = 0.0
    if (x == 0.0 || y == 0.0) return 0.0 end
    a0 = min(a,b)
    if (a0 < 8.0) # go to 100

        if (x <= 0.375)# go to 10
            lnx = log(x)
            lny = dlnrel(-x)
        elseif (y <= 0.375)# go to 20
            lnx = dlnrel(-y)
            lny = log(y)
        else
            lnx = log(x)
            lny = log(y) 
        end

        z = a*lnx + b*lny
        if (a0 >= 1.0) # go to 30
            z -= dbetln(a,b)
            return desum(mu, z)
        end
            
#-----------------------------------------------------------------------
#              PROCEDURE FOR A .LT. 1 OR B .LT. 1 
#-----------------------------------------------------------------------
        b0 = max(a,b)
        #if (b0 .ge. 8.0) go to 80
        if (b0 <= 1.0) # go to 60

#                   algorithm for b0 .le. 1

            brcompval = exp(mu + z)
            if (brcompval == 0.0) return 0.0 end

            apb = a + b
            if (apb <= 1.0) # go to 40
                z = 1.0 + rgamma1pm1(apb)
            else
                u = a + b - 1.0
                z = (1.0 + rgamma1pm1(u))/apb 
            end

            c = (1.0 + rgamma1pm1(a))*(1.0 + rgamma1pm1(b))/z
            return brcompval*(a0*c)/(1.0 + a0/b0)
        elseif (b0 < 8.0)

#                ALGORITHM FOR 1 .LT. B0 .LT. 8

            u = lgamma1p(a0)
            n = itrunc(b0 - 1.0)
            if (n > 1) # go to 70
                c = 1.0
                for i = 1:n 
                    b0 -= 1.0
                    c *= b0/(a0 + b0)
                end
                u += log(c)
            end

            z -= u
            b0 -= 1.0 
            apb = a0 + b0 
            if (apb <= 1.0)
                t = 1.0 + rgamma1pm1(apb)
            else
                u = a0 + b0 - 1.0
                t = (1.0 + rgamma1pm1(u))/apb
            end
            return a0*exp(mu + z)*(1.0 + rgamma1pm1(b0))/t
        else

#                   ALGORITHM FOR B0 .GE. 8

            u = lgamma1p(a0) + dlgdiv(a0,b0)
            return a0*exp(mu + z - u)
        end
    end
#-----------------------------------------------------------------------
#              PROCEDURE FOR A .GE. 8 AND B .GE. 8
#-----------------------------------------------------------------------
    if (a <= b) # go to 101 
        h = a/b
        x0 = h/(1.0 + h)
        y0 = 1.0/(1.0 + h)
        lambda = a - (a + b)*x
    else
        h = b/a
        x0 = 1.0/(1.0 + h)
        y0 = h/(1.0 + h)
        lambda = (a + b)*y - b
    end
    e = -lambda/a 
    if (abs(e) <= 0.6) # go to 111
        u = -log1pmx(e)
    else
        u = e - log(x/x0)
    end

    e = lambda/b
    if (abs(e) <= 0.6) # go to 121
        v = -log1pmx(e)
    else
        v = e - log(y/y0)
    end
    z = exp(mu - (a*u + b*v))
    return cnst*sqrt(b*x0)*z*exp(-dbcorr(a,b))
end 

function bgrat(a::Real, b::Real, x::Real, y::Real, w::Real, precision::Real)
#     SUBROUTINE BGRAT(A, B, X, Y, W, EPS, IERR)
#-----------------------------------------------------------------------
#     ASYMPTOTIC EXPANSION FOR IX(A,B) WHEN A IS LARGER THAN B.
#     THE RESULT OF THE EXPANSION IS ADDED TO W. IT IS ASSUMED
#     THAT A .GE. 15 AND B .LE. 1.  EPS IS THE TOLERANCE USED.
#     IERR IS A VARIABLE THAT REPORTS THE STATUS OF THE RESULTS.
#-----------------------------------------------------------------------
#      REAL J, L, LNX, NU, N2
#      REAL C(30), D(30)

    c = Array(Float64, 30)
    d = Array(Float64, 30)

    bm1 = (b - 0.5) - 0.5
    nu = a + 0.5*bm1
    if y <= 0.375
        lnx = dlnrel(-y)
    else
        lnx = log(x)
    end
    z = -nu*lnx
    if (b*z == 0.0) return w end # How should errors be handled? They are ognored in the original progam. ("Cannot calculate expansion")

#                 COMPUTATION OF THE EXPANSION
#                 SET R = EXP(-Z)*Z**B/GAMMA(B)

    r = b*(1.0 + rgamma1pm1(b))*z^b
    r *= exp(a*lnx)*exp(0.5*bm1*lnx) 
    u = dlgdiv(b,a) + b*log(nu)
    u = r*exp(-u) 
    if (u == 0.0) return w end # How should errors be handled? They are ognored in the original progam. ("Cannot calculate expansion")
    p,q = grat1(b,z,r,precision)

    v = 0.25*(1.0/nu)^2
    t2 = 0.25*lnx*lnx
    l = w/u
    j = q/r
    sumval = j
    t = 1.0
    cn = 1.0
    n2 = 0.0
    for n = 1:30
        bp2n = b + n2
        j = (bp2n*(bp2n + 1.0)*j + (z + bp2n + 1.0)*t)*v
        n2 += 2.0
        t *= t2
        cn /= n2*(n2 + 1.0)
        c[n] = cn
        s = 0.0
        if (n > 1) 
            nm1 = n - 1
            coef = b - n
            for i = 1:nm1
                s += coef*c[i]*d[n-i] 
                coef += b
            end
        end
        d[n] = bm1*cn + s/n
        dj = d[n]*j
        sumval += dj
        if (sumval <= 0.0) return w end # How should errors be handled? They are ognored in the original progam. ("Cannot calculate expansion")
        if (abs(dj) <= precision*(sumval + l)) break end
    end

#                  ADD THE RESULTS TO W

    return w + u*sumval
end

function grat1(a::Real,x::Real,r::Real,precision::Real)
#     SUBROUTINE GRAT1 (A,X,R,P,Q,EPS)
#     REAL J, L
#-----------------------------------------------------------------------
#        EVALUATION OF THE INCOMPLETE GAMMA RATIO FUNCTIONS 
#                      P(A,X) AND Q(A,X)
#
#     IT IS ASSUMED THAT A .LE. 1.  EPS IS THE TOLERANCE TO BE USED.
#     THE INPUT ARGUMENT R HAS THE VALUE E**(-X)*X**A/GAMMA(A).
#-----------------------------------------------------------------------
    if (a*x == 0.0) return (x <= a ? (0.0, 1.0) : (1.0, 0.0)) end
    if (a == 0.5)
        if (x < 0.25) # go to 121
            p = erf(sqrt(x))
            return p, 1.0 - p
            return
        else
            q = erfc(sqrt(x))
            return 1.0 - q, q
        end
    end
    if (x < 1.1) # go to 10

#             TAYLOR SERIES FOR P(A,X)/X**A

        an = 3.0
        c = x
        sumval = x/(a + 3.0)
        tol = 0.1*precision/(a + 1.0)
        t = c
        while (abs(t) > tol)
           an += 1.0
           c *= -(x/an)
           t = c/(a + an)
           sumval += t
        end
        j = a*x*((sumval/6.0 - 0.5/(a + 2.0))*x + 1.0/(a + 1.0)) 
        
        z = a*log(x) 
        h = rgamma1pm1(a)
        g = 1.0 + h
        while true
            if (x >= 0.25)
                if (a < x/2.59) break end
            else 
                if (z > -.13394) break end
            end

            w = exp(z)
            p = w*g*(1.0 - j)
            return p, 1.0 - p
            break
        end

        l = expm1(z)
        w = 0.5 + (0.5 + l)
        q = (w*j - l)*g - h
        if (q < 0.0) return 1.0, 0.0 end
        return 1.0 - q, q

#              CONTINUED FRACTION EXPANSION

    else
        a2nm1 = 1.0
        a2n = 1.0
        b2nm1 = x
        b2n = x + (1.0 - a)
        c = 1.0
        am0 = a2nm1/b2nm1
        an0 = a2n/b2n
        while (abs(an0 - am0) >= precision*an0)
           a2nm1 = x*a2n + c*a2nm1
           b2nm1 = x*b2n + c*b2nm1
           am0 = a2nm1/b2nm1
           c = c + 1.0
           cma = c - a
           a2n = a2nm1 + cma*a2n
           b2n = b2nm1 + cma*b2n
           an0 = a2n/b2n
        end
        q = r*an0
        return 1.0 - q, q
    end
end

function basym(a::Real, b::Real, lambda::Real, precision::Real)
#     REAL FUNCTION BASYM(A, B, LAMBDA, EPS)
#-----------------------------------------------------------------------
#     ASYMPTOTIC EXPANSION FOR IX(A,B) FOR LARGE A AND B.
#     LAMBDA = (A + B)*Y - B  AND EPS IS THE TOLERANCE USED.
#     IT IS ASSUMED THAT LAMBDA IS NONNEGATIVE AND THAT
#     A AND B ARE GREATER THAN OR EQUAL TO 15.
#-----------------------------------------------------------------------
#     REAL J0, J1, LAMBDA
#------------------------
#     ****** NUM IS THE MAXIMUM VALUE THAT N CAN TAKE IN THE DO LOOP
#            ENDING AT STATEMENT 50. IT IS REQUIRED THAT NUM BE EVEN. 
#            THE ARRAYS A0, B0, C, D HAVE DIMENSION NUM + 1.
#
    num = 20
    a0 = Array(Float64, num + 1)
    b0 = Array(Float64, num + 1)
    c = Array(Float64, num + 1)
    d = Array(Float64, num + 1) 

#------------------------
#     E0 = 2/SQRT(PI)
#     E1 = 2**(-3/2)
#------------------------
    e0 = 1.1283791670955126
    e1 = 0.3535533905932737
#------------------------
    basymval = 0.0
    if (a < b) # go to 10
        h = a/b
        r0 = 1.0/(1.0 + h)
        r1 = (b - a)/b
        w0 = 1.0/sqrt(a*(1.0 + h))
    else
        h = b/a
        r0 = 1.0/(1.0 + h)
        r1 = (b - a)/a
        w0 = 1.0/sqrt(b*(1.0 + h))
    end

    f = -a*log1pmx(-lambda/a) - b*log1pmx(lambda/b)
    t = exp(-f)
    if (t == 0.0) return basymval end
    z0 = sqrt(f)
    z = 0.5*(z0/e1)
    z2 = f + f

    a0[1] = (2.0/3.0)*r1
    c[1] = - 0.5*a0[1]
    d[1] = - c[1]
    j0 = (0.5/e0)*erfcx(z0)
    j1 = e1
    sumval = j0 + d[1]*w0*j1
    s = 1.0
    h2 = h*h
    hn = 1.0
    w = w0
    znm1 = z
    zn = z2
    for n = 2:2:num
        hn = h2*hn 
        a0[n] = 2.0*r0*(1.0 + h*hn)/(n + 2.0)
        np1 = n + 1
        s += hn 
        a0[np1] = 2.0*r1*s/(n + 3.0)

        for i = n:np1
            r = -0.5*(i + 1.0)
            b0[1] = r*a0[1]
            for m = 2:i
                bsum = 0.0
                mm1 = m - 1
                for j = 1:mm1
                    mmj = m - j
                    bsum += (j*r - mmj)*a0[j]*b0[mmj]
                end
                b0[m] = r*a0[m] + bsum/m
            end
            c[i] = b0[i]/(i + 1.0)

            dsum = 0.0 
            im1 = i - 1
            for j = 1:im1
                imj = i - j
                dsum += d[imj]*c[j]
            end
            d[i] = -(dsum + c[i])
        end

        j0 = e1*znm1 + (n - 1.0)*j0
        j1 = e1*zn + n*j1
        znm1 *= z2
        zn *= z2 
        w *= w0
        t0 = d[n]*w*j0
        w *= w0
        t1 = d[np1]*w*j1
        sumval += t0 + t1
        if ((abs(t0) + abs(t1)) <= precision*sumval) break end
    end

    u = exp(-dbcorr(a,b))
    return e0*t*u*sumval
end


function dbcorr(a0::Real, b0::Real)
#-----------------------------------------------------------------------
#
#     EVALUATION OF DEL(A) + DEL(B0) - DEL(A) + B0) WHERE
#     LN(GAMMA(X)) = (X - 0.5)*LN(X) - X + 0.5*LN(2*PI) + DEL(X).
#     IT IS ASSUMED THAT A0 .GE. 10 AND B0 .GE. 10.
#
#                         --------
#
#     THE SERIES FOR DEL(X), WHICH APPLIES FOR X .GE. 10, WAS
#     DERIVED BY A.H. MORRIS FROM THE CHEBYSHEV SERIES IN THE
#     SLATEC LIBRARY OBTAINED BY WAYNE FULLERTON (LOS ALAMOS).
#
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A0, B0
#     DOUBLE PRECISION A, B, C, E(15), H, S(15), T, W, X, X2, Z
    s = Array(Float64, 15)
#--------------------------
    e=[.833333333333333333333333333333e-01,
      -.277777777777777777777777752282e-04,
       .793650793650793650791732130419e-07,
      -.595238095238095232389839236182e-09,
       .841750841750832853294451671990e-11,
      -.191752691751854612334149171243e-12,
       .641025640510325475730918472625e-14,
      -.295506514125338232839867823991e-15,
       .179643716359402238723287696452e-16,
      -.139228964661627791231203060395e-17,
       .133802855014020915603275339093e-18,
      -.154246009867966094273710216533e-19,
       .197701992980957427278370133333e-20,
      -.234065664793997056856992426667e-21,
       .171348014966398575409015466667e-22]
#--------------------------
    a = min(a0, b0)
    b = max(a0, b0)

    h = a/b
    c = h/(1.0 + h)
    x = 1.0/(1.0 + h)
    x2 = x*x
#
#        COMPUTE (1 - X**N)/(1 - X) FOR N = 1,3,5,...
#            STORE THESE VALUES IN S(1),S(2),...
#
    s[1] = 1.0
    for j = 1:14
        s[j+1] = 1.0 + (x + x2*s[j])
    end
#
#                SET W = DEL(B) - DEL(A + B)
#
    t = (10.0/b)^2
    w = e[15]*s[15]
    for j = 1:14
        k = 15 - j
        w = t*w + e[k]*s[k]
    end
    w *= c/b
#
#                    COMPUTE  DEL(A) + W
#
    t = (10.0/a)^2
    z = e[15]
    for j = 1:14
        k = 15 - j
        z = t*z + e[k]
    end
    return z/a + w
end

function dlgdiv(a::Real, b::Real)
#-----------------------------------------------------------------------
#
#     COMPUTATION OF LN(GAMMA(B)/GAMMA(A+B)) FOR B .GE. 10
#
#                         --------
#
#     DLGDIV USES A SERIES FOR THE FUNCTION DEL(X) WHERE
#     LN(GAMMA(X)) = (X - 0.5)*LN(X) - X + 0.5*LN(2*PI) + DEL(X).
#     THE SERIES FOR DEL(X), WHICH APPLIES FOR X .GE. 10, WAS
#     DERIVED BY A.H. MORRIS FROM THE CHEBYSHEV SERIES IN THE
#     SLATEC LIBRARY OBTAINED BY WAYNE FULLERTON (LOS ALAMOS).
#
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A, B
#     DOUBLE PRECISION C, D, E(15), H, S(15), T, U, V, W, X, X2
    s = Array(Float64, 15)
#     DOUBLE PRECISION DLNREL
#--------------------------
    e=[.833333333333333333333333333333e-01
      -.277777777777777777777777752282e-04
       .793650793650793650791732130419e-07
      -.595238095238095232389839236182e-09
       .841750841750832853294451671990e-11
      -.191752691751854612334149171243e-12
       .641025640510325475730918472625e-14
      -.295506514125338232839867823991e-15
       .179643716359402238723287696452e-16
      -.139228964661627791231203060395e-17      
       .133802855014020915603275339093e-18
      -.154246009867966094273710216533e-19
       .197701992980957427278370133333e-20
      -.234065664793997056856992426667e-21
       .171348014966398575409015466667e-22]
#--------------------------
    if (a > b) # go to 10
        h = b/a
        c = 1.0/(1.0 + h)
        x = h/(1.0 + h)
        d = a + (b - 0.5)
    else
        h = a/b
        c = h/(1.0 + h)
        x = 1.0/(1.0 + h)
        d = b + (a - 0.50)
    end
#
#        COMPUTE (1 - X**N)/(1 - X) FOR N = 1,3,5,...
#            STORE THESE VALUES IN S(1),S(2),...
#
    x2 = x*x
    s[1] = 1.0
    for j = 1:14
        s[j + 1] = 1.0 + (x + x2*s[j])
    end
#
#                SET W = DEL(B) - DEL(A + B)
#
    t = (10.0/b)^2
    w = e[15]*s[15]
    for j = 1:14
        k = 15 - j
        w = t*w + e[k]*s[k]
    end
    w *= c/b
#
#                    COMBINE THE RESULTS
#
    u = d*dlnrel(a/b)
    v = a*(log(b) - 1.0)
    if u > v # go to 40
        return (w - v) - u
    end
    return (w - u) - v
end

function dlnrel(a::Real)
#-----------------------------------------------------------------------
#            EVALUATION OF THE FUNCTION LN(1 + A)
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A, T, T2, W, Z
#     DOUBLE PRECISION P0, P1, P2, P3, Q1, Q2, Q3, Q4
#     DOUBLE PRECISION C1, C2, C3, C4, C5
#-------------------------
    p0 =  .7692307692307692307680e-01
    p1 = -.1505958055914600184836e+00
    p2 =  .9302355725278521726994e-01
    p3 = -.1787900022182327735804e-01
    q1 = -.2824412139355646910683e+01
    q2 =  .2892424216041495392509e+01
    q3 = -.1263560605948009364422e+01
    q4 =  .1966769435894561313526e+00
#-------------------------
#     CI = 1/(2I + 1)
#-------------------------
    c1 = .3333333333333333333333333333333e+00
    c2 = .2000000000000000000000000000000e+00
    c3 = .1428571428571428571428571428571e+00
    c4 = .1111111111111111111111111111111e+00
    c5 = .9090909090909090909090909090909e-01
#-------------------------
    if abs(a) >= 0.375 # go to 10
        t = 1.0 + a
        if a < 0.0 t = 0.50 + (0.50 + a) end
        return log(t)
    end
#
#        W IS A MINIMAX APPROXIMATION OF THE SERIES
#
#               C6 + C7*T**2 + C8*T**4 + ...
#
#        THIS APPROXIMATION IS ACCURATE TO WITHIN
#        1.6 UNITS OF THE 21-ST SIGNIFICANT DIGIT.
#        THE RESULTING VALUE FOR 1.D0 + T2*Z IS
#        ACCURATE TO WITHIN 1 UNIT OF THE 30-TH
#        SIGNIFICANT DIGIT.
#
    t = a/(a + 2.0)
    t2 = t*t
    w = (((p3*t2 + p2)*t2 + p1)*t2 + p0)/((((q4*t2 + q3)*t2 + q2)*t2 + q1)*t2 + 1.0)

    z = ((((w*t2 + c5)*t2 + c4)*t2 + c3)*t2 + c2)*t2 + c1
    return 2.0*t*(1.0 + t2*z)
end

function alnrel(a::Float32)
#-----------------------------------------------------------------------
#            EVALUATION OF THE FUNCTION LN(1 + A)
#-----------------------------------------------------------------------
    p1 = -.129418923021993f+01
    p2 =  .405303492862024f+00
    p3 = -.178874546012214f-01
    q1 = -.162752256355323f+01
    q2 =  .747811014037616f+00
    q3 = -.845104217945565f-01
#--------------------------
    if abs(a) <= 0.375 # go to 10
        t = a/(a + 2.0f0)
        t2 = t*t
        w = (((p3*t2 + p2)*t2 + p1)*t2 + 1.0f0)/(((q3*t2 + q2)*t2 + q1)*t2 + 1.0f0)
        return 2.0f0*t*w
    end

    x = 1.0f0 + a
    if a < 0.0 x = (a + 0.5f0) + 0.5f0 end
    return log(x)
end

function dbetln(a0::Real, b0::Real)
#-----------------------------------------------------------------------
#     EVALUATION OF THE LOGARITHM OF THE BETA FUNCTION
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A0, B0
#     DOUBLE PRECISION A, B, C, E, H, SN, U, V, W, Z
#     DOUBLE PRECISION DBCORR, DGAMLN, DGSMLN, DLGDIV, DLNREL
#--------------------------
#     E = 0.5*LN(2*PI)
#--------------------------
    e = .9189385332046727417803297364056
#--------------------------
    a = min(a0,b0)
    b = max(a0,b0)
    if a < 10.0 # go to 60
        if a < 1.0 # go to 20
#-----------------------------------------------------------------------
#                   PROCEDURE WHEN A .LT. 1
#-----------------------------------------------------------------------
            if b < 10.0 # go to 10
                return lgamma(a) + (lgamma(b) - lgamma(a + b))
            else
                return lgamma(a) + dlgdiv(a,b)
            end
        end
#-----------------------------------------------------------------------
#               PROCEDURE WHEN 1 .LE. A .LT. 10
#-----------------------------------------------------------------------
        while true
            if a <= 2.0 # go to 30
                if b <= 2.0 # go to 21
                    return lgamma(a) + lgamma(b) - dgsmln(a,b)
                end
                w = 0.0
                if b < 10.0 break end # go to 40
                return lgamma(a) + dlgdiv(a,b)
            end
#
#               REDUCTION OF A WHEN B .LE. 1000
#
            if b > 1.0e3 # go to 50
#
#               REDUCTION OF A WHEN B .GT. 1000
#
                n = itrunc(a - 1.0)
                w = 1.0
                for i = 1:n
                    a -= 1.0
                    w *= a/(1.0 + a/b)
                end
                sn = n
                return (log(w) - sn*log(b)) + (lgamma(a) + dlgdiv(a,b))
            end

            n = itrunc(a - 1.0)
            w = 1.0
            for i = 1:n
                a -= 1.0
                h = a/b
                w *= h/(1.0 + h)
            end
            w = log(w)
            if b < 10.0 break end # go to 40
            return w + lgamma(a) + dlgdiv(a,b)
        end
#
#                REDUCTION OF B WHEN B .LT. 10
#
        n = b - 1.0
        z = 1.0
        for i = 1:n
            b -= 1.0
            z *= b/(a + b)
        end
        return w + log(z) + (lgamma(a) + (lgamma(b) - dgsmln(a,b)))
    end
#-----------------------------------------------------------------------
#                  PROCEDURE WHEN A .GE. 10
#-----------------------------------------------------------------------
    w = dbcorr(a,b)
    h = a/b
    c = h/(1.0 + h)
    u = -(a - 0.50)*log(c)
    v = b*dlnrel(h)
    if u > v # go to 61
        return (((-0.5*log(b) + e) + w) - v) - u
    end
    return (((-0.5*log(b) + e) + w) - u) - v
end

function dgsmln(a::Real, b::Real)
#-----------------------------------------------------------------------
#          EVALUATION OF THE FUNCTION LN(GAMMA(A + B))
#          FOR 1 .LE. A .LE. 2  AND  1 .LE. B .LE. 2
#-----------------------------------------------------------------------
#     DOUBLE PRECISION A, B, X
#     DOUBLE PRECISION DGMLN1, DLNREL

    x = (a - 1.0) + (b - 1.0)
    if x <= 0.50 # go to 10
        return lgamma1p(1.0 + x)
    end
    if x < 1.50 # go to 20
        return lgamma1p(x) + dlnrel(x)
    end
    return lgamma1p(x - 1.0) + log(x*(1.0 + x))
end

function desum(mu::Integer, x::Real)
#-----------------------------------------------------------------------
#                    EVALUATION OF EXP(MU + X)
#-----------------------------------------------------------------------
#     DOUBLE PRECISION X, W
#
    if x <= 0.0 # go to 10

        if mu < 0 return exp(mu)*exp(x) end
        w = mu + x
        if w > 0.0 return exp(mu)*exp(x) end
        return exp(w)
    end

    if mu > 0 return exp(mu)*exp(x) end
    w = mu + x
    if w < 0.0 return exp(mu)*exp(x) end
    return exp(w)
end

# NSWC DGAM1
function rgamma1pm1(x::Float64)
#-----------------------------------------------------------------------
#     EVALUATION OF 1/GAMMA(1 + X) - 1  FOR -0.5 .LE. X .LE. 1.5
#-----------------------------------------------------------------------
#     DOUBLE PRECISION X, D, T, W, Z
#     DOUBLE PRECISION A0, A1, B1, B2, B3, B4, B5, B6, B7, B8
#     DOUBLE PRECISION P0, P1, P2, P3, P4, P5, P6, Q1, Q2, Q3, Q4
#     DOUBLE PRECISION C, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9,
#    *                 C10, C11, C12, C13
#----------------------------
    a0 =  .611609510448141581788e-08
    a1 =  .624730830116465516210e-08    
    b1 =  .203610414066806987300e+00
    b2 =  .266205348428949217746e-01
    b3 =  .493944979382446875238e-03
    b4 = -.851419432440314906588e-05
    b5 = -.643045481779353022248e-05
    b6 =  .992641840672773722196e-06
    b7 = -.607761895722825260739e-07
    b8 =  .195755836614639731882e-09
#----------------------------
    p0 =  .6116095104481415817861e-08
    p1 =  .6871674113067198736152e-08
    p2 =  .6820161668496170657918e-09
    p3 =  .4686843322948848031080e-10
    p4 =  .1572833027710446286995e-11
    p5 = -.1249441572276366213222e-12
    p6 =  .4343529937408594255178e-14
    q1 =  .3056961078365221025009e+00
    q2 =  .5464213086042296536016e-01
    q3 =  .4956830093825887312020e-02
    q4 =  .2692369466186361192876e-03
#----------------------------
#     c = c0 - 1
#----------------------------
    c  = -.422784335098467139393487909917598e+00
#----------------------------
    c0  =  .577215664901532860606512090082402e+00
    c1  = -.655878071520253881077019515145390e+00
    c2  = -.420026350340952355290039348754298e-01
    c3  =  .166538611382291489501700795102105e+00
    c4  = -.421977345555443367482083012891874e-01
    c5  = -.962197152787697356211492167234820e-02
    c6  =  .721894324666309954239501034044657e-02
    c7  = -.116516759185906511211397108401839e-02
    c8  = -.215241674114950972815729963053648e-03
    c9  =  .128050282388116186153198626328164e-03    
    c10 = -.201348547807882386556893914210218e-04
    c11 = -.125049348214267065734535947383309e-05
    c12 =  .113302723198169588237412962033074e-05
    c13 = -.205633841697760710345015413002057e-06
#----------------------------
    t = x
    d = x - 0.5
    if d > 0.0 t = d - 0.5 end
    if t == 0 # 40,10,20
        return 0.0
    elseif t > 0
#------------
#
#                CASE WHEN 0 .LT. T .LE. 0.5
#
#              W IS A MINIMAX APPROXIMATION FOR
#              THE SERIES A(15) + A(16)*T + ...
#
#------------
        w = ((((((p6*t + p5)*t + p4)*t + p3)*t + p2)*t + p1)*t + p0)/((((q4*t + q3)*t + q2)*t + q1)*t + 1.0)     
        z = (((((((((((((w*t + c13)*t + c12)*t + c11)*t + c10)*t + c9)*t + c8)*t + c7)*t + c6)*t + c5)*t + c4)*t + c3)*t + c2)*t + c1)*t + c0

        if d <= 0.0 # go to 30
            return x*z
        end
        return (t/x)*((z - 0.5) - 0.5)
    end
#------------
#
#                CASE WHEN -0.5 .LE. T .LT. 0
#
#              W IS A MINIMAX APPROXIMATION FOR
#              THE SERIES A(15) + A(16)*T + ...
#
#------------
    w = (a1*t + a0)/((((((((b8*t + b7)*t + b6)*t + b5)*t + b4)*t + b3)*t + b2)*t + b1)*t + 1.0)
    z = (((((((((((((w*t + c13)*t + c12)*t + c11)*t + c10)*t + c9)*t + c8)*t + c7)*t + c6)*t + c5)*t + c4)*t + c3)*t + c2)*t + c1)*t + c

    if d <= 0.0 # go to 50
        return x*((z + 0.5) + 0.5)
    end
    return t*z/x
end

# NSWC GAM1
function rgamma1pm1(a::Float32)
#-----------------------------------------------------------------------
#     COMPUTATION OF 1/GAMMA(A+1) - 1  FOR -0.5 .LE. A .LE. 1.5
#-----------------------------------------------------------------------
      # REAL P(7), Q(5), R(9)
#------------------------
    t = a
    d = a - 0.5f0
    if d > 0.0f0 t = d - 0.5f0 end
    if t == 0 # 30,10,20
        return 0.0f0
    elseif t < 0
        top = @horner(t, .577215664901533f+00, 
                        -.409078193005776f+00, 
                        -.230975380857675f+00, 
                         .597275330452234f-01,
                         .766968181649490f-02, 
                        -.514889771323592f-02,
                         .589597428611429f-03)
        bot = @horner(t,.100000000000000f+01, 
                        .427569613095214f+00,
                        .158451672430138f+00, 
                        .261132021441447f-01,
                        .423244297896961f-02)
        w = top/bot
        if d <= 0.0 # go to 21
            return a*w
        end
        return (t/a)*((w - 0.5f0) - 0.5f0)
    end

    top = @horner(t, -.422784335098468f+00, 
                     -.771330383816272f+00,
                     -.244757765222226f+00, 
                      .118378989872749f+00,
                      .930357293360349f-03, 
                     -.118290993445146f-01,
                      .223047661158249f-02, 
                      .266505979058923f-03,
                     -.132674909766242f-03)
    bot = @horner(t, 1.0f0, 
                      .273076135303957f+00, 
                      .559398236957378f-01)
    w = top/bot
    if d <= 0.0 # go to 31
        return a*((w + 0.5f0) + 0.5f0)
    end
    return t*w/a
end

function lgamma1p(x)
    if -0.5 <= x <= 1.5
        return -log1p(rgamma1pm1(x))
    else
        lgamma(one(x)+x)
    end
end

### End of regularized incomplete beta function

# Multidimensional gamma / partial gamma function
function lpgamma(p::Int64, a::Float64)
    res::Float64 = p * (p - 1.0) / 4.0 * log(pi)
    for ii in 1:p
        res += lgamma(a + (1.0 - ii) / 2.0)
    end
    return res
end
