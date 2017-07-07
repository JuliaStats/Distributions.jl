using Documenter, Distributions

makedocs(
    format = :html,
    sitename = "Distributions.jl",
    modules = [Distributions],
    pages = ["index.md",
             "starting.md",
             "types.md",
             "univariate.md",
             "truncate.md",
             "multivariate.md",
             "matrix.md",
             "mixture.md",
             "fit.md",
             "extends.md"]
)

deploydocs(
    repo = "github.com/JuliaStats/Distributions.jl.git",
    target = "build",
    julia  = "0.6",
    deps = nothing,
    make = nothing
)
