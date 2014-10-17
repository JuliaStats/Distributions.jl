
immutable GammaRmathSampler <: Sampleable{Univariate,Continuous}
    α::Float64
    scale::Float64
end

rand(s::GammaRmathSampler) = 
    ccall((:rgamma, "libRmath-julia"), Float64, (Float64, Float64), s.α, s.scale)



# "Generating gamma variates by a modified rejection technique"
# J.H. Ahrens, U. Dieter
# Communications of the ACM, Vol 25(1), 1982, pp 47-54
# doi:10.1145/358315.358390

# suitable for scale >= 1.0

immutable GammaGDSampler <: Sampleable{Univariate,Continuous}
    a::Float64
    s2::Float64
    s::Float64
    d::Float64
    q0::Float64
    b::Float64
    σ::Float64
    c::Float64
    scale::Float64
end

function GammaGDSampler(a::Float64,scale::Float64)
    # Step 1
    s2 = a-0.5
    s = sqrt(s2)
    d = 5.656854249492381 - 12.0s # 4*sqrt(2) - 12s

    # Step 4
    ia = 1.0/a
    q0 = ia*@horner(ia,
                    0.0416666664,
                    0.0208333723,
                    0.0079849875,
                    0.0015746717,
                    -0.0003349403,
                    0.0003340332,
                    0.0006053049,
                    -0.0004701849,
                    0.0001710320)

    if a <= 3.686
        b = 0.463 + s + 0.178s2
        σ = 1.235
        c = 0.195/s - 0.079 + 0.16s
    elseif a <= 13.022
        b = 1.654 + 0.0076s2
        σ = 1.68/s + 0.275
        c = 0.062/s + 0.024
    else
        b = 1.77
        σ = 0.75
        c = 0.1515/s
    end

    GammaGDSampler(a,s2,s,d,q0,b,σ,c,scale)
end

function rand(s::GammaGDSampler)
    # Step 2
    t = randn()
    x = s.s + 0.5t
    t >= 0.0 && return x*x*s.scale

    # Step 3
    u = rand()
    s.d*u <= t*t*t && return x*x*s.scale

    # Step 5
    if x > 0.0
        # Step 6
        v = t/(2.0*s.s)
        if abs(v) > 0.25
            q = s.q0 - s.s*t + 0.25*t*t + 2.0*s.s2*log1p(v)
        else
            q = s.q0 + 0.5*t*t*(v*@horner(v,
                                         0.333333333,
                                         -0.249999949,
                                         0.199999867,
                                         -0.1666774828,
                                         0.142873973,
                                         -0.124385581,
                                         0.110368310,
                                         -0.112750886,
                                         0.10408986))
        end

        # Step 7
        log1p(-u) <= q && return x*x*s.scale
    end

    # Step 8
    @label step8
    e = Base.Random.randmtzig_exprnd()
    u = 2.0rand() - 1.0
    t = s.b + e*s.σ*sign(u)

    # Step 9
    t < -0.718_744_837_717_19 && @goto step8

    # Step 10
    v = t/(2.0*s.s)
    if abs(v) > 0.25
        q = s.q0 - s.s*t + 0.25*t*t + 2.0*s.s2*log1p(v)
    else
        q = s.q0 + 0.5*t*t*(v*@horner(v,
                                      0.333333333,
                                      -0.249999949,
                                      0.199999867,
                                      -0.1666774828,
                                      0.142873973,
                                      -0.124385581,
                                      0.110368310,
                                      -0.112750886,
                                      0.10408986))
    end

    # Step 11
    (q <= 0.0 || s.c*abs(u) > expm1(q)*exp(e-0.5t*t)) && @goto step8

    # Step 12
    x = s.s+0.5t
    return x*x*s.scale
end

#  A simple method for generating gamma variables - Marsaglia and Tsang (2000)
#  http://www.cparity.com/projects/AcmClassification/samples/358414.pdf
#  Page 369
#  basic simulation loop for pre-computed d and c
#

immutable GammaMTSampler <: Sampleable{Univariate,Continuous}
    iα::Float64
    d::Float64
    c::Float64
    κ::Float64
end

function GammaMTSampler(α::Float64, scale::Float64)
    if α >= 1.0
        iα = 1.0
        d = α - 1/3
    else
        iα = 1.0 / α
        d = α + 2/3
    end
    c = 1.0 / sqrt(9.0 * d)
    κ = d * scale
    GammaMTSampler(iα, d, c, κ)
end

GammaMTSampler(α::Float64) = GammaMTSampler(α, 1.0)

function rand(s::GammaMTSampler)
    d = s.d
    c = s.c
    iα = s.iα

    v = 0.0
    while true
        x = randn()
        v = 1.0 + c * x
        while v <= 0.0
            x = randn()
            v = 1.0 + c * x
        end
        v *= (v * v)
        u = rand()
        x2 = x * x
        if u < 1.0 - 0.331 * abs2(x2)
            break
        end
        if log(u) < 0.5 * x2 + d * (1.0 - v + log(v))
            break
        end
    end
    v *= s.κ
    if iα > 1.0
        v *= (rand() ^ iα)
    end
    return v
end

