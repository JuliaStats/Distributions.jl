
# Discrete univariate

convert(::Type{Binomial}, d::Bernoulli) = Binomial(1, d.p)

# Continuous univariate

convert(::Type{Gamma}, d::Exponential) = Gamma(1.0, d.θ)
convert(::Type{Gamma}, d::Erlang) = Gamma(d.α, d.θ)

convert(::Type{GeneralizedGamma}, d::Gamma) = GeneralizedGamma(d.θ, d.α, 1.0)
convert(::Type{GeneralizedGamma}, d::Exponential) = GeneralizedGamma(d.θ, 1.0, 1.0)
convert(::Type{GeneralizedGamma}, d::Weibull) = GeneralizedGamma(d.θ, d.α, d.α)
