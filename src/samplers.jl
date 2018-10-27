# delegation of samplers

for fname in ["aliastable.jl",
              "binomial.jl",
              "poissonbinomial.jl",
              "poisson.jl",
              "exponential.jl",
              "gamma.jl",
              "multinomial.jl",
              "vonmises.jl",
              "vonmisesfisher.jl",
              "discretenonparametric.jl",
              "categorical.jl"]

    include(joinpath("samplers", fname))
end
