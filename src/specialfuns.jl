# Special functions

# See:
#   Martin Maechler (2012) "Accurately Computing log(1 − exp(− |a|))"
#   http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

# log(1-exp(x)) 
# NOTE: different than Maechler (2012), no negation inside parantheses
log1mexp(x::Real) = x >= -0.6931471805599453 ? log(-expm1(x)) : log1p(-exp(x))
# log(1+exp(x))
log1pexp(x::Real) = log1p(exp(x))
log1pexp(x::Float64) = x <= 18.0 ? log1p(exp(x)) : x <= 33.3 ? x + exp(-x) : x
log1pexp(x::Float32) = x <= 9f0 ? log1p(exp(x)) : x <= 16f0 ? x + exp(-x) : x
log1pexp(x::Integer) = log1pexp(float(x))
# log(exp(x)-1)
logexpm1(x::Real) = log(expm1(x))
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
