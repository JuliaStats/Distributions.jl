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
        "censored.md",
        "multivariate.md",
        "matrix.md",
        "cholesky.md",
        "mixture.md",
        "fit.md",
        "extends.md",
        "density_interface.md",
    ]
)

deploydocs(;
    repo = "github.com/JuliaStats/Distributions.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"],
    push_preview=true,
)
