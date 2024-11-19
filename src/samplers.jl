# delegation of samplers

"""
    rand!(::AbstractRNG, ::Sampleable, ::AbstractArray)

Samples in-place from the sampler and stores the result in the provided array.
"""
rand!(::AbstractRNG, ::Sampleable, ::AbstractArray)

"""
    rand(::AbstractRNG, ::Sampleable)

Samples from the sampler and returns the result.
"""
rand(::AbstractRNG, ::Sampleable)

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
              "categorical.jl",
              "productnamedtuple.jl",
             ]

    include(joinpath("samplers", fname))
end
