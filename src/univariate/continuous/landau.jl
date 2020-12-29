using StaticArrays
"""
    Landau(μ,θ)

The *Landau distribution* with location `μ` and scale `θ` has probability density function

```math
p(x) = \\frac{1}{2 \\pi i} \\int_{\\mu-i\\infty}^{\\mu+i\\infty} e^{s \\log(s) + x s}\\, ds
```

```julia
Landau()       # Landau distribution with zero location and unit scale, i.e. Landau(0, 1)
Landau(u)      # Landau distribution with location u and unit scale, i.e. Landau(u, 1)
Landau(u, b)   # Landau distribution with location u ans scale b

params(d)       # Get the parameters, i.e. (u, b)
location(d)     # Get the location parameter, i.e. u
scale(d)        # Get the scale parameter, i.e. b
```

External links

* [Landau distribution on Wikipedia](http://en.wikipedia.org/wiki/Landau_distribution)

"""
struct Landau{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    θ::T
    Landau{T}(µ::T, θ::T) where {T} = new{T}(µ, θ)
end


function Landau(μ::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Landau, θ > zero(θ))
    return Landau{T}(μ, θ)
end

Landau(μ::Real, θ::Real) = Landau(promote(μ, θ)...)
Landau(μ::Integer, θ::Integer) = Landau(float(μ), float(θ))
Landau(μ::T) where {T <: Real} = Landau(μ, one(T))
Landau() = Landau(0.0, 1.0, check_args=false)

@distr_support Landau -Inf Inf

#### Conversions
function convert(::Type{Landau{T}}, μ::S, θ::S) where {T <: Real, S <: Real}
    Landau(T(μ), T(θ))
end
function convert(::Type{Landau{T}}, d::Landau{S}) where {T <: Real, S <: Real}
    Landau(T(d.μ), T(d.θ), check_args=false)
end

#### Parameters

location(d::Landau) = d.μ
scale(d::Landau) = d.θ

params(d::Landau) = (d.μ, d.θ)
@inline partype(d::Landau{T}) where {T<:Real} = T


#### Evaluation

pdf(d::Landau, x::Real) = _landau_pdf((x-d.μ)/d.θ)

cdf(d::Landau, x::Real) = _landau_cdf((x-d.μ)/d.θ)

function cf(d::Landau, t::Real)
    exp(im*t*d.μ - 2im*d.θ*t/π*log(abs(t)) - d.θ*abs(t))
end

# https://root.cern.ch/doc/v622/PdfFuncMathCore_8cxx_source.html
function _landau_pdf(x, xi=1, x0=0)
    # landau pdf : algorithm from cernlib g110 denlan
    p1 = @SVector [0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253]
    q1 = @SVector [1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063]

    p2 = @SVector [0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211]
    q2 = @SVector [1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714]

    p3 = @SVector [0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101]
    q3 = @SVector [1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675]

    p4 = @SVector [0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186]
    q4 = @SVector [1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511]

    p5 = @SVector [1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910]
    q5 = @SVector [1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357]

    p6 = @SVector [1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109]
    q6 = @SVector [1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939]

    a1 = @SVector [0.04166666667,-0.01996527778, 0.02709538966]

    a2 = @SVector [-1.845568670,-4.284640743]

    xi <= 0 && return 0
    v = (x - x0)/xi
    u = ue = us = denlan = 0.0
    if v < -5.5
        u   = exp(v+1.0)
        u < 1e-10 && return 0.0
        ue  = exp(-1/u)
        us  = sqrt(u)
        denlan = 0.3989422803*(ue/us)*(1+(a1[1]+(a1[2]+a1[3]*u)*u)*u)
    elseif(v < -1)
        u   = exp(-v-1)
        denlan = exp(-u)*sqrt(u)*
        (p1[1]+(p1[2]+(p1[3]+(p1[4]+p1[5]*v)*v)*v)*v)/
        (q1[1]+(q1[2]+(q1[3]+(q1[4]+q1[5]*v)*v)*v)*v)
    elseif(v < 1)
        denlan = (p2[1]+(p2[2]+(p2[3]+(p2[4]+p2[5]*v)*v)*v)*v)/
        (q2[1]+(q2[2]+(q2[3]+(q2[4]+q2[5]*v)*v)*v)*v)
    elseif(v < 5)
        denlan = (p3[1]+(p3[2]+(p3[3]+(p3[4]+p3[5]*v)*v)*v)*v)/
        (q3[1]+(q3[2]+(q3[3]+(q3[4]+q3[5]*v)*v)*v)*v)
    elseif(v < 12)
        u   = 1/v
        denlan = u*u*(p4[1]+(p4[2]+(p4[3]+(p4[4]+p4[5]*u)*u)*u)*u)/
        (q4[1]+(q4[2]+(q4[3]+(q4[4]+q4[5]*u)*u)*u)*u)
    elseif(v < 50)
        u   = 1/v
        denlan = u*u*(p5[1]+(p5[2]+(p5[3]+(p5[4]+p5[5]*u)*u)*u)*u)/
        (q5[1]+(q5[2]+(q5[3]+(q5[4]+q5[5]*u)*u)*u)*u)
    elseif(v < 300)
        u   = 1/v
        denlan = u*u*(p6[1]+(p6[2]+(p6[3]+(p6[4]+p6[5]*u)*u)*u)*u)/
        (q6[1]+(q6[2]+(q6[3]+(q6[4]+q6[5]*u)*u)*u)*u)
    else
        u   = 1/(v-v*log(v)/(v+1))
        denlan = u*u*(1+(a2[1]+a2[2]*u)*u)
    end
    return denlan/xi
end 

#https://root.cern.ch/doc/v622/ProbFuncMathCore_8cxx_source.html
function _landau_cdf(x, xi=1, x0=0)
    # implementation of landau distribution (from DISLAN)
    #The algorithm was taken from the Cernlib function dislan(G110)
    #Reference: K.S.Kolbig and B.Schorr, "A program package for the Landau
    #distribution", Computer Phys.Comm., 31(1984), 97-111

    p1 = @SVector [0.2514091491e+0,-0.6250580444e-1, 0.1458381230e-1,-0.2108817737e-2, 0.7411247290e-3]
    q1 = @SVector [1.0            ,-0.5571175625e-2, 0.6225310236e-1,-0.3137378427e-2, 0.1931496439e-2]

    p2 = @SVector [0.2868328584e+0, 0.3564363231e+0, 0.1523518695e+0, 0.2251304883e-1]
    q2 = @SVector [1.0            , 0.6191136137e+0, 0.1720721448e+0, 0.2278594771e-1]

    p3 = @SVector [0.2868329066e+0, 0.3003828436e+0, 0.9950951941e-1, 0.8733827185e-2]
    q3 = @SVector [1.0            , 0.4237190502e+0, 0.1095631512e+0, 0.8693851567e-2]

    p4 = @SVector [0.1000351630e+1, 0.4503592498e+1, 0.1085883880e+2, 0.7536052269e+1]
    q4 = @SVector [1.0            , 0.5539969678e+1, 0.1933581111e+2, 0.2721321508e+2]

    p5 = @SVector [0.1000006517e+1, 0.4909414111e+2, 0.8505544753e+2, 0.1532153455e+3]
    q5 = @SVector [1.0            , 0.5009928881e+2, 0.1399819104e+3, 0.4200002909e+3]

    p6 = @SVector [0.1000000983e+1, 0.1329868456e+3, 0.9162149244e+3,-0.9605054274e+3]
    q6 = @SVector [1.0            , 0.1339887843e+3, 0.1055990413e+4, 0.5532224619e+3]

    a1 = @SVector [0              ,-0.4583333333e+0, 0.6675347222e+0,-0.1641741416e+1]
    a2 = @SVector [0              , 1.0            ,-0.4227843351e+0,-0.2043403138e+1]

    v = (x - x0)/xi

    if (v < -5.5)
        u   = exp(v+1)
        lan = 0.3989422803*exp(-1. /u)*sqrt(u)*(1+(a1[2]+(a1[3]+a1[4]*u)*u)*u)
    elseif (v < -1 )
        u   = exp(-v-1)
        lan = (exp(-u)/sqrt(u))*(p1[1]+(p1[2]+(p1[3]+(p1[4]+p1[5]*v)*v)*v)*v)/
        (q1[1]+(q1[2]+(q1[3]+(q1[4]+q1[5]*v)*v)*v)*v)
    elseif (v < 1)
        lan = (p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)/(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)
    elseif (v < 4)
        lan = (p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)/(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)
    elseif (v < 12)
        u   = 1. /v
        lan = (p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)/(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)
    elseif (v < 50)
        u   = 1. /v
        lan = (p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)/(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)
    elseif (v < 300)
        u   = 1. /v
        lan = (p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)/(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)
    else
        u   = 1. /(v-v*log(v)/(v+1))
        lan = 1-(a2[2]+(a2[3]+a2[4]*u)*u)*u
    end
    return lan
end
