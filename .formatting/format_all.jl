using Runic

project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")

println("Formatting code with Runic...")

# Format all files in the project
not_formatted = Runic.main(["--inplace", project_path])
if not_formatted == 0
    @info "Formatting completed successfully."
else
    @warn "Formatting failed!"
end
exit(not_formatted)
