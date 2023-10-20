module TestUtils

import ..Distributions
import ..Distributions.Random

"""
    test_mvnormal(
        g::AbstractMvNormal,
        n_tsamples::Int=10^6,
        rng::Union{Random.AbstractRNG, Nothing}=nothing,
    )

Test that `AbstractMvNormal` implements the expected API.

!!! Note
    On Julia >= 1.9, you have to load the `Test` standard library to be able to use
    this function.
"""
test_mvnormal(
    g::Distributions.AbstractMvNormal,
    n_tsamples::Int=10^6,
    rng::Union{Random.AbstractRNG, Nothing}=nothing
)

if isdefined(Base, :get_extension) && isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        # Better error message if users forget to load Test
        Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
            if exc.f === test_mvnormal &&
                (Base.get_extension(Distributions, :DistributionsTestExt) === nothing)
                print(io, "\nDid you forget to load Test?")
            end
        end
    end
end

end
