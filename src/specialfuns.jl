# Special functions
import Base.Math.@horner

# See:
#   Martin Maechler (2012) "Accurately Computing log(1 − exp(− |a|))"
#   http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

# log(1-exp(x)) 
# NOTE: different than Maechler (2012), no negation inside parantheses
log1mexp(x::Real) = x >= loghalf ? log(-expm1(x)) : log1p(-exp(x))
# log(1+exp(x))
log1pexp(x::Real) = x > zero(x) ? x + log1p(exp(-x)) : log1p(exp(x))
log1pexp(x::Float64) = x <= 18.0 ? log1p(exp(x)) : x <= 33.3 ? x + exp(-x) : x
log1pexp(x::Float32) = x <= 9f0 ? log1p(exp(x)) : x <= 16f0 ? x + exp(-x) : x
log1pexp(x::Integer) = log1pexp(float(x))
# log(exp(x)-1)
logexpm1(x::Real) = x + log1mexp(-x)
logexpm1(x::Float64) = x <= 18.0 ? log(expm1(x)) : x <= 33.3 ? x - exp(-x) : x
logexpm1(x::Float32) = x <= 9f0 ? log(expm1(x)) : x <= 16f0 ? x - exp(-x) : x
logexpm1(x::Integer) = logexpm1(float(x))

# log(1+x^2)
log1psq(x::FloatingPoint) = (ax = abs(x); ax < maxintfloat(x) ? log1p(ax*ax) : 2*log(ax))

# log(1+x)-x
# accurate ~2ulps for -0.227 < x < 0.315
function log1pmx_kernel(x::Float64)
    r = x/(x+2.0)
    t = r*r
    w = @horner(t,
                6.66666666666666667e-1, # 2/3
                4.00000000000000000e-1, # 2/5
                2.85714285714285714e-1, # 2/7
                2.22222222222222222e-1, # 2/9
                1.81818181818181818e-1, # 2/11
                1.53846153846153846e-1, # 2/13
                1.33333333333333333e-1, # 2/15
                1.17647058823529412e-1) # 2/17
    hxsq = 0.5*x*x
    r*(hxsq+w*t)-hxsq
end

# use naive calculation or range reduction outside kernel range.
# accurate ~2ulps for all x
function log1pmx(x::Float64)
    if !(-0.7 < x < 0.9)
        return log1p(x) - x
    elseif x > 0.315
        u = (x-0.5)/1.5
        return log1pmx_kernel(u) - 9.45348918918356180e-2 - 0.5*u
    elseif x > -0.227
        return log1pmx_kernel(x)
    elseif x > -0.4
        u = (x+0.25)/0.75
        return log1pmx_kernel(u) - 3.76820724517809274e-2 + 0.25*u
    elseif x > -0.6
        u = (x+0.5)*2.0
        return log1pmx_kernel(u) - 1.93147180559945309e-1 + 0.5*u
    else
        u = (x+0.625)/0.375
        return log1pmx_kernel(u) - 3.55829253011726237e-1 + 0.625*u
    end
end

# log(x) - x + 1
function logmxp1(x::Float64)
    if x <= 0.3
        return (log(x) + 1.0) - x
    elseif x <= 0.4
        u = (x-0.375)/0.375
        return log1pmx_kernel(u) - 3.55829253011726237e-1 + 0.625*u
    elseif x <= 0.6
        u = 2.0*(x-0.5)
        return log1pmx_kernel(u) - 1.93147180559945309e-1 + 0.5*u
    else
        return log1pmx(x-1.0)
    end
end



φ(z::Real) = exp(-0.5*z*z)/√2π
logφ(z::Real) = -0.5*(z*z + log2π)

Φ(z::Real) = 0.5*erfc(-z/√2)
Φc(z::Real) = 0.5*erfc(z/√2)
logΦ(z::Real) = z < -1.0 ? log(0.5*erfcx(-z/√2)) - 0.5*z*z : log1p(-0.5*erfc(z/√2))
logΦc(z::Real) = z > 1.0 ? log(0.5*erfcx(z/√2)) - 0.5*z*z : log1p(-0.5*erfc(-z/√2))

# Rational approximations for the inverse cdf, from:
#   Wichura, M.J. (1988) Algorithm AS 241: The Percentage Points of the Normal Distribution
#   Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 37, No. 3, pp. 477-484
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

        function $fn($arg::Real)
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


# Multidimensional gamma / partial gamma function
function lpgamma(p::Int, a::Float64)
    res::Float64 = p * (p - 1.0) / 4.0 * log(pi)
    for ii in 1:p
        res += lgamma(a + (1.0 - ii) / 2.0)
    end
    return res
end



# Remainder term after Stirling's approximation to the log-gamma function
# lstirling(x) = lgamma(x) + x - (x-0.5)*log(x) - 0.5*log2π
#              = 1/(12x) - 1/(360x^3) + 1/(1260x^5) + ...
# Asymptotic expansion from:
#   Temme, N. (1996) Special functions: An introduction to the classical
#   functions of mathematical physics, Wiley, New York, ISBN: 0-471-11313-1,
#   Chapter 3.6, pp 61-65.
# Relative error of approximation is bounded by
#   (174611/125400 x^-19) / (1/12 x^-1 - 1/360 x^-3)
# which is < 1/2 ulp for x >= 10.0
# total numeric error appears to be < 2 ulps
lstirling_asym(x::Integer) = lstirling_asym(float(x))

function lstirling_asym(x::Float64)
    t = 1.0/(x*x)
    @horner(t,
             8.33333333333333333e-2, #  1/12 x^-1
            -2.77777777777777778e-3, # -1/360 x^-3
             7.93650793650793651e-4, #  1/1260 x^-5
            -5.95238095238095238e-4, # -1/1680 x^-7
             8.41750841750841751e-4, #  1/1188 x^-9
            -1.91752691752691753e-3, # -691/360360 x^-11
             6.41025641025641026e-3, #  1/156 x^-13
            -2.95506535947712418e-2, # -3617/122400 x^-15
             1.79644372368830573e-1)/x #  43867/244188 x^-17
end

function lstirling_asym(x::Float32)
    t = 1f0/(x*x)
    @horner(t,
             8.333333333333f-2, #  1/12 x^-1
            -2.777777777777f-3, # -1/360 x^-3
             7.936507936508f-4, #  1/1260 x^-5
            -5.952380952381f-4, # -1/1680 x^-7
             8.417508417508f-4)/x #  1/1188 x^-9
end
