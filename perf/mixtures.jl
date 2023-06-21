using BenchmarkTools: @btime
import Random
using Distributions: AbstractMixtureModel, MixtureModel, LogNormal, Normal, pdf, ncomponents, probs, component, components, ContinuousUnivariateDistribution
using Test

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
        else
            zero(eltype(p))
        end
    end
end

function improved_no_inbound(d, x)
    p = probs(d)
    return sum(enumerate(p)) do (i, pi)
        if pi > 0
           c = component(d, i)
           pdf(c, x) * pi
        else
            zero(eltype(p))
        end
    end
end

function ifelse_version(d, x)
    ps = probs(d)
    cs = components(d)
    P = eltype(ps)
    return sum(ifelse(ps[i] > 0, ps[i] * pdf(cs[i], x), zero(P)) for i in eachindex(ps))
end

function forloop(d, x)
    ps = probs(d)
    cs = components(d)
    s = zero(eltype(ps))
    @inbounds for i in eachindex(ps)
        if ps[i] > 0
            s += ps[i] * pdf(cs[i], x)
        end
    end
    return s
end

function indexed_sum_comp(d, x)
    ps = probs(d)
    cs = components(d)
    @inbounds sum(ps[i] * pdf(cs[i], x) for i in eachindex(ps) if ps[i] > 0)
end

function indexed_boolprod(d, x)
    ps = probs(d)
    cs = components(d)
    @inbounds sum((ps[i] > 0) * (ps[i] * pdf(cs[i], x)) for i in eachindex(ps))
end

function indexed_boolprod_noinbound(d, x)
    ps = probs(d)
    cs = components(d)
    return sum((ps[i] > 0) * (ps[i] * pdf(cs[i], x)) for i in eachindex(ps))
end

function sumcomp_cond(d, x)
    ps = probs(d)
    cs = components(d)
    s = zero(eltype(ps))
    @inbounds sum(ps[i] * pdf(cs[i], x) for i in eachindex(ps) if ps[i] > 0)
end

distributions = [
    Normal(-1.0, 0.3),
    Normal(0.0, 0.5),
    Normal(3.0, 1.0),
]

priors = [0.25, 0.25, 0.5]

gmm_normal = MixtureModel(distributions, priors)

@info "Small Gaussian mixture"

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
    @info "ifelse_version"
    v3 = @btime ifelse_version($gmm_normal, $x)
    @info "forloop"
    v4 = @btime forloop($gmm_normal, $x)
    @info "indexed_sum_comp"
    v5 = @btime indexed_sum_comp($gmm_normal, $x)
    @info "indexed_boolprod"
    v6 = @btime indexed_boolprod($gmm_normal, $x)
    @info "indexed_boolprod_noinbound"
    v6 = @btime indexed_boolprod_noinbound($gmm_normal, $x)
    @info "sumcomp_cond"
    v7 = @btime sumcomp_cond($gmm_normal, $x)
    @assert vman ≈ vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4 ≈ v5 ≈ v6 ≈ v7
    @info "==================="
end

large_normals = [Normal(rand(), rand()) for _ in 1:1000]
large_probs = [rand() for _ in 1:1000]
large_probs .= large_probs ./ sum(large_probs)

gmm_large = MixtureModel(large_normals, large_probs)

@info "Large Gaussian mixture"

Random.seed!(42)
for x in rand(5)
    @info "sampling $x"
    @info "evaluate_manual_pdf"
    vman = @btime evaluate_manual_pdf($large_normals, $large_probs, $x)
    @info "current_master"
    vmaster =  @btime current_master($gmm_large, $x)
    @info "improved_version"
    v1 = @btime improved_version($gmm_large, $x)
    @info "improved_no_inbound"
    v2 = @btime improved_no_inbound($gmm_large, $x)
    @info "ifelse_version"
    v3 = @btime ifelse_version($gmm_large, $x)
    @info "forloop"
    v4 = @btime forloop($gmm_large, $x)
    @info "indexed_sum_comp"
    v5 = @btime indexed_sum_comp($gmm_large, $x)
    @info "indexed_boolprod"
    v6 = @btime indexed_boolprod($gmm_large, $x)
    @info "indexed_boolprod_noinbound"
    v6 = @btime indexed_boolprod_noinbound($gmm_large, $x)
    @info "sumcomp_cond"
    v7 = @btime sumcomp_cond($gmm_large, $x)
    @assert vman ≈ vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4 ≈ v5 ≈ v6 ≈ v7
    @info "==================="
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
    vmaster =  @btime current_master($gmm_het, $x)
    @info "improved_version"
    v1 = @btime improved_version($gmm_het, $x)
    @info "improved_no_inbound"
    v2 = @btime improved_no_inbound($gmm_het, $x)
    @info "ifelse_version"
    v3 = @btime ifelse_version($gmm_het, $x)
    @info "forloop"
    v4 = @btime forloop($gmm_het, $x)
    @info "indexed_sum_comp"
    v5 = @btime indexed_sum_comp($gmm_het, $x)
    @info "indexed_boolprod"
    v6 = @btime indexed_boolprod($gmm_het, $x)
    @info "indexed_boolprod_noinbound"
    v6 = @btime indexed_boolprod_noinbound($gmm_het, $x)
    @info "sumcomp_cond"
    v7 = @btime sumcomp_cond($gmm_het, $x)
    @assert vman ≈ vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4 ≈ v5 ≈ v6 ≈ v7
    @info "==================="
end


@info "Test with one NaN"

distributions = [
    Normal(-1.0, 0.3),
    Normal(0.0, 0.5),
    Normal(3.0, 1.0),
    Normal(NaN, 1.0),
]

priors = [0.25, 0.25, 0.5, 0.0]

gmm_normal = MixtureModel(distributions, priors)

Random.seed!(42)
for x in rand(1)
    @info "sampling $x"
    @info "current_master"
    vmaster =  @btime current_master($gmm_normal, $x)
    @info "improved_version"
    v1 = @btime improved_version($gmm_normal, $x)
    @info "improved_no_inbound"
    v2 = @btime improved_no_inbound($gmm_normal, $x)
    @info "ifelse_version"
    v3 = @btime ifelse_version($gmm_normal, $x)
    @info "forloop"
    v4 = @btime forloop($gmm_normal, $x)
    @info "indexed_sum_comp"
    v5 = @btime indexed_sum_comp($gmm_normal, $x)
    @info "indexed_boolprod"
    v6 = @btime indexed_boolprod($gmm_normal, $x)
    @info "indexed_boolprod_noinbound"
    v7 = @btime indexed_boolprod_noinbound($gmm_normal, $x)
    @info "sumcomp_cond"
    v8 = @btime sumcomp_cond($gmm_normal, $x)
    @test vmaster ≈ v1 ≈ v2 ≈ v3 ≈ v4 ≈ v5 ≈ v6 ≈ v7 ≈ v8
end
