
# Discrete univariate

convert(::Type{Binomial}, d::Bernoulli) = Binomial(1, d.p)

# Continuous univariate

convert(::Type{Gamma}, d::Exponential) = Gamma(1.0, d.θ)
convert(::Type{Gamma}, d::Erlang) = Gamma(d.α, d.θ)
convert(::Type{Stable}, d::Normal) = Stable(2, 0, d.σ/√2, d.μ)
convert(::Type{Stable}, d::Cauchy) = Stable(1, 0, d.σ, d.μ)
convert(::Type{Stable}, d::Levy) = Stable(1/2, 1, d.σ, d.μ)
