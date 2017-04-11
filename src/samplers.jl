# delegation of samplers

for fname in ["categorical.jl",
              "binomial.jl",
              "poissonbinomial.jl",
              "poisson.jl",
              "exponential.jl",
              "gamma.jl",
              "multinomial.jl",
              "vonmises.jl",
              "vonmisesfisher.jl"]

    include(joinpath("samplers", fname))
end
