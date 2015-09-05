
# Discrete univariate

convert(::Type{Binomial}, d::Bernoulli) = Binomial(1, d.p)

# Continuous univariate

convert(::Type{Gamma}, d::Exponential) = Gamma(1.0, d.θ)
convert(::Type{Gamma}, d::Erlang) = Gamma(d.α, d.θ)
