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
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)
