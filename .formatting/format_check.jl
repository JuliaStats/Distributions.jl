using Runic

project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")

println("Checking code formatting with Runic...")

# Check if files are properly formatted
not_formatted = Runic.main(["--check", "--diff", project_path])

if not_formatted == 0
    println("✅ All files are properly formatted!")
    exit(0)
else
    println("❌ Formatting check failed!")
    println("Some files are not properly formatted.")
    println("To fix formatting, run: julia --project=.formatting -e 'using Pkg; Pkg.instantiate(); include(\".formatting/format_all.jl\")'")
    exit(1)
end
