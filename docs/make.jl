using Documenter, Distributions
import Random: AbstractRNG, rand!

makedocs(
    sitename = "Distributions.jl",
    modules  = [Distributions],
    doctest  = false,
    pages    = [
        "index.md",
        "starting.md",
        "types.md",
        "univariate.md",
        "truncate.md",
        "multivariate.md",
        "matrix.md",
        "mixture.md",
        "fit.md",
        "extends.md",
    ]
)

deploydocs(
    repo = "github.com/JuliaStats/Distributions.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)
