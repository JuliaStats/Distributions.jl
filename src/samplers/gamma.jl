
immutable GammaRmathSampler <: Sampleable{Univariate,Continuous}
    α::Float64
    scale::Float64
end

rand(s::GammaRmathSampler) = 
    ccall((:rgamma, "libRmath-julia"), Float64, (Float64, Float64), s.α, s.scale)

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

