
struct VonMisesSampler <: Sampleable{Univariate,Continuous}
    μ::Float64
    κ::Float64
    r::Float64

    function VonMisesSampler(μ::Float64, κ::Float64)
        τ = 1.0 + sqrt(1.0 + 4 * abs2(κ))
        ρ = (τ - sqrt(2.0 * τ)) / (2.0 * κ)
        new(μ, κ, (1.0 + abs2(ρ)) / (2.0 * ρ))
    end
end

# algorithm from
#     DJ Best & NI Fisher (1979). Efficient Simulation of the von Mises
#     Distribution. Journal of the Royal Statistical Society. Series C
#     (Applied Statistics), 28(2), 152-157.
function rand(rng::AbstractRNG, s::VonMisesSampler)
    f = 0.0
    local x::Float64
    if s.κ > 700.0
        x = s.μ + randn(rng) / sqrt(s.κ)
    else
        while true
            t, u = 0.0, 0.0
            while true
                d = abs2(rand(rng) - 0.5)
                e = abs2(rand(rng) - 0.5)
                if d + e <= 0.25
                    t = d / e
                    u = 4 * (d + e)
                    break
                end
            end
            z = (1.0 - t) / (1.0 + t)
            f = (1.0 + s.r * z) / (s.r + z)
            c = s.κ * (s.r - f)
            if c * (2.0 - c) > u || log(c / u) + 1 >= c
                break
            end
        end
        acf = acos(f)
        x = s.μ + (rand(rng, Bool) ? acf : -acf)
    end
    return x
end
