# Sampler for drawing random number of a Gamma distribution

# Routines for sampling from Gamma distribution

#  A simple method for generating gamma variables - Marsaglia and Tsang (2000)
#  http://www.cparity.com/projects/AcmClassification/samples/358414.pdf
#  Page 369
#  basic simulation loop for pre-computed d and c

const _v13 = 1.0 / 3.0

# a sampler for Gamma(α)
immutable GammaSampler
    α::Float64
    iα::Float64  # iα = α > 1 ? 1.0 : 1.0 / α
    d::Float64
    c::Float64

    function GammaSampler(α::Float64) 
        local iα::Float64, d::Float64
        if α > 1.0 
            iα = 1.0
            d = α - _v13
        else
            iα = 1.0 / α
            d = α + 1.0 - _v13
        end 
        new(α, iα, d, 1.0 / sqrt(9.0 * d))
    end
end


function rand(s::GammaSampler)
    d::Float64 = s.d
    c::Float64 = s.c

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
        x2 = x^2
        if u < 1.0 - 0.331 * x2^2
            break
        end
        if log(u) < 0.5 * x2 + d * (1.0 - v + log(v))
            break
        end
    end
    v *= d

    if s.α <= 1.0
        v *= (rand()^s.iα)
    end
    return v::Float64
end

randg(α::Float64) = rand(GammaSampler(α))
randg(α::Real) = rand(float64(α))


