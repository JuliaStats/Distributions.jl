# special log functions

# log1mexp and log1pexp are based on:
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
# still inaccurate for values close to log(2)
logexpm1(x::BigFloat) = x <= realmaxexp(typeof(x)) ? log(expm1(x)) : x 
logexpm1(x::Float64) = x <= 18.0 ? log(expm1(x)) : x <= 33.3 ? x - exp(-x) : x
logexpm1(x::Float32) = x <= 9f0 ? log(expm1(x)) : x <= 16f0 ? x - exp(-x) : x
logexpm1(x::Integer) = logexpm1(float(x))



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


log1pmx(x) = log1p(x) - x
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

# NSWC RLOG1
function log1pmx(x::Float32)
    a = 0.566749439387324f-01
    b = 0.456512608815524f-01

    if x >= -0.39f0 && x <= 0.57f0 # go to 100
        if x < -0.18f0 # go to 10
            u = (x + 0.3f0)/0.7f0
            up2 = u + 2f0
            w1 = a - u*0.3f0
        elseif x > 0.18 # go to 20
            t = 0.75f0*x
            u = t - 0.25f0
            up2 = t + 1.75f0
            w1 = b + u/3f0
        else
            u = x
            up2 = u + 2f0
            w1 = 0f0
        end
#
#                  SERIES EXPANSION
#
        r = u/up2
        t = r*r

        w = @horner(t,
                    0.333333333333333f+00, 
                    -.224696413112536f+00,
                    0.620886815375787f-02) /
        @horner(t, 1f0,
                -.127408923933623f+01,
                0.354508718369557f+00)

        return r*(2f0*t*w - u) - w1
    end
    return log1p(x) - x
end
