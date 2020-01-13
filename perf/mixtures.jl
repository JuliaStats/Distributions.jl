using BenchmarkTools: @btime
import Random
using Distributions: AbstractMixtureModel, MixtureModel, LogNormal, Normal, pdf, ncomponents, probs, component, components, ContinuousUnivariateDistribution

# v0.22.1
function current_master(d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K
    v = 0.0
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            v += pdf(c, x) * pi
        end
    end
    return v
end

# compute the overhead of having a mixture
function evaluate_manual_pdf(distributions, priors, x)
    return sum(pdf(d, x) * p for (d, p) in zip(distributions, priors))
end

function improved_version(d, x)
   p = probs(d)
   return sum(enumerate(p)) do (i, pi)
       if pi > 0
           @inbounds c = component(d, i)
           pdf(c, x) * pi
       end
   end
end

function improved_no_inbound(d, x)
   p = probs(d)
   return sum(enumerate(p)) do (i, pi)
       if pi > 0
           c = component(d, i)
           pdf(c, x) * pi
       end
   end
end

function improved_zipped(d, x)
   ps = probs(d)
   cs = components(d)
   return sum(p * pdf(c, x) for (p, c) in zip(ps, cs))
end

function improved_indexed(d, x)
   ps = probs(d)
   cs = components(d)
   return sum(ps[i] * pdf(cs[i], x) for i in eachindex(ps))
end

distributions = [
    Normal(-1.0, 0.3),
    Normal(0.0, 0.5),
    Normal(3.0, 1.0),
]

priors = [0.25, 0.25, 0.5]

gmm_normal = MixtureModel(distributions, priors)

@info "Small Gausian mixture"

Random.seed!(42)
for x in rand(5)
    @info "sampling $x"
    @info "evaluate_manual_pdf"
    vman = @btime evaluate_manual_pdf($distributions, $priors, $x)
    @info "current_master"
    vmaster =  @btime current_master($gmm_normal, $x)
    @info "improved_version"
    v1 = @btime improved_version($gmm_normal, $x)
    @info "improved_no_inbound"
    v2 = @btime improved_no_inbound($gmm_normal, $x)
    @info "improved_zipped"
    v3 = @btime improved_zipped($gmm_normal, $x)
    @info "improved_indexed"
    v4 = @btime improved_indexed($gmm_normal, $x)
    @assert vman ≈ vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4
end

large_normals = [Normal(rand(), rand()) for _ in 1:1000]
large_probs = [rand() for _ in 1:1000]
large_probs .= large_probs ./ sum(large_probs)

gmm_large = MixtureModel(large_normals, large_probs)

@info "Large Gausian mixture"

Random.seed!(42)
for x in rand(5)
    @info "sampling $x"
    @info "evaluate_manual_pdf"
    vman = @btime evaluate_manual_pdf($large_normals, $large_probs, $x)
    @info "current_master"
    vmaster = @btime current_master($gmm_large, $x)
    @info "improved_version"
    v1 = @btime improved_version($gmm_large, $x)
    @info "improved_no_inbound"
    v2 = @btime improved_no_inbound($gmm_large, $x)
    @info "improved_zipped"
    v3 = @btime improved_zipped($gmm_large, $x)
    @info "improved_indexed"
    v4 = @btime improved_indexed($gmm_large, $x)
    @assert vman ≈ vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4
end

large_het = append!(
    ContinuousUnivariateDistribution[Normal(rand(), rand()) for _ in 1:1000],
    ContinuousUnivariateDistribution[LogNormal(rand(), rand()) for _ in 1:1000],
)

large_het_probs = [rand() for _ in 1:2000]
large_het_probs .= large_het_probs ./ sum(large_het_probs)

gmm_het = MixtureModel(large_het, large_het_probs)

@info "Heterogeneous distributions"

Random.seed!(42)
for x in rand(5)
    @info "sampling $x"
    @info "evaluate_manual_pdf"
    vman = @btime evaluate_manual_pdf($large_het, $large_het_probs, $x)
    @info "current_master"
    vmaster = @btime current_master($gmm_het, $x)
    @info "improved_version"
    v1 = @btime improved_version($gmm_het, $x)
    @info "improved_no_inbound"
    v2 = @btime improved_no_inbound($gmm_het, $x)
    @info "improved_zipped"
    v3 = @btime improved_zipped($gmm_het, $x)
    @info "improved_indexed"
    v4 = @btime improved_indexed($gmm_het, $x)
    @assert vman ≈ vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4
end
