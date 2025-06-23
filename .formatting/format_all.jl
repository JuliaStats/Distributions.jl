using JuliaFormatter

project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")

not_formatted = format(project_path; verbose = true)
if not_formatted
    @info "Formatting verified."
else
    @warn "Formatting verification failed: Some files are not properly formatted!"
end
exit(not_formatted ? 0 : 1)
