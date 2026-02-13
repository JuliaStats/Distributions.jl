using JuliaFormatter

project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")

println("Formatting code with JuliaFormatter...")

format(project_path; verbose = true)
println("Formatting completed successfully.")
