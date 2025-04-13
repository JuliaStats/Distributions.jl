using Documenter, Distributions
import Random: AbstractRNG, rand!

makedocs(;
    sitename = "Distributions.jl",
    modules  = [Distributions],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages    = [
        "index.md",
        "starting.md",
        "types.md",
        "univariate.md",
        "truncate.md",
        "censored.md",
        "multivariate.md",
        "matrix.md",
        "reshape.md",
        "cholesky.md",
        "mixture.md",
        "product.md",
        "order_statistics.md",
        "convolution.md",
        "fit.md",
        "extends.md",
        "density_interface.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo = "github.com/JuliaStats/Distributions.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"],
    push_preview = true,
)
