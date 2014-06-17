# Sampler for drawing random number of a Gamma distribution

function sampler(d::Gamma)
    d.shape > 1.0 ? GammaMTSampler(d) : 
    d.shape == 1.0 ? Exponential(d.scale) :
    d.shape == 0.5 ? NormalSqSampler(d) :
    GammaMTPowerSampler(d)
end

rand(d::Gamma) = rand(sampler(d))
rand!(d::Gamma,a::Array) = rand!(sampler(d),a)

sampler(d::Chisq) = sampler(Gamma(0.5*d.df,2.0))
rand(d::Chisq) = rand(sampler(d))
rand!(d::Chisq,a::Array) = rand!(sampler(d),a)

immutable NormalSqSampler <: Sampler{Univariate,Continuous}
    scale::Float64
end

NormalSqSampler(d::Gamma) = NormalSqSampler(0.5*d.scale)

rand(s::NormalSqSampler) = s.scale*(randn())^2



#  A simple method for generating gamma variables - Marsaglia and Tsang (2000)
#  http://www.cparity.com/projects/AcmClassification/samples/358414.pdf
#  Page 369
#  basic simulation loop for pre-computed d and c
immutable GammaMTSampler <: Sampler{Univariate,Continuous}
    d::Float64
    c::Float64
    scale::Float64
end

function GammaMTSampler(g::Gamma)
    d = g.shape - 1.0/3.0
    GammaMTSampler(d, 1.0 / sqrt(9.0 * d), g.scale * d)
end

# for shape < 1.0: see note on page 371.
immutable GammaMTPowerSampler <: Sampler{Univariate,Continuous}
    gmt::GammaMTSampler
    iα::Float64
end

function GammaMTPowerSampler(g::Gamma)
    d = g.shape + 2.0/3.0
    GammaMTPowerSampler(GammaMTSampler(d, 1.0 / sqrt(9.0 * d), g.scale * d), 1.0/g.shape)
end
    

function rand(s::GammaMTSampler)
    v = 0.0
    while true
        x = randn()
        v = 1.0 + s.c * x
        while v <= 0.0
            x = randn()
            v = 1.0 + s.c * x
        end
        v *= (v * v)
        u = rand()
        x2 = x^2
        if u < 1.0 - 0.331 * x2^2
            break
        end
        if log(u) < 0.5 * x2 + s.d * (1.0 - v + log(v))
            break
        end
    end
    v * s.scale
end

rand(s::GammaMTPowerSampler) = rand(s.gmt) * rand()^s.iα
