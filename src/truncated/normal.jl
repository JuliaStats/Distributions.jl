# Truncated normal distribution

TruncatedNormal(mu::Float64, sigma::Float64, a::Float64, b::Float64) =
    Truncated(Normal(mu, sigma), a, b)

TruncatedNormal(mu::Real, sigma::Real, a::Real, b::Real) =
    @compat TruncatedNormal(Float64(mu), Float64(sigma), Float64(a), Float64(b))

### statistics

minimum(d::Truncated{Normal}) = d.lower
maximum(d::Truncated{Normal}) = d.upper


function mode(d::Truncated{Normal})
    μ = mean(d.untruncated)
    d.upper < μ ? d.upper :
    d.lower > μ ? d.lower : μ
end

modes(d::Truncated{Normal}) = [mode(d)]


function mean(d::Truncated{Normal})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    μ + ((φ(a) - φ(b)) / d.tp) * σ
end

function var(d::Truncated{Normal})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    z = d.tp
    φa = φ(a)
    φb = φ(b)
    aφa = isinf(a) ? 0.0 : a * φa
    bφb = isinf(b) ? 0.0 : b * φb
    t1 = (aφa - bφb) / z
    t2 = abs2((φa - φb) / z)
    abs2(σ) * (1 + t1 - t2)
end

function entropy(d::Truncated{Normal})
    d0 = d.untruncated
    z = d.tp
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    aφa = isinf(a) ? 0.0 : a * φ(a)
    bφb = isinf(b) ? 0.0 : b * φ(b)
    0.5 * (log2π + 1.) + log(σ * z) + (aφa - bφb) / (2.0 * z)
end


### sampling

## Benchmarks doesn't seem to show that this specialized
## sampler is faster than the generic quantile-based method

# function rand(d::Truncated{Normal})
#     d0 = d.untruncated
#     μ = mean(d0)
#     σ = std(d0)
#     a = (d.lower - μ) / σ
#     b = (d.upper - μ) / σ
#     z = randnt(a, b, d.tp)
#     return μ + σ * z
# end

# Rejection sampler based on algorithm from Robert (1995)
#
#  - Available at http://arxiv.org/abs/0907.4010
#
# function randnt(lb::Float64, ub::Float64, tp::Float64)
#     r::Float64
#     if tp > 0.3   # has considerable chance of falling in [lb, ub]
#         r = randn()
#         while r < lb || r > ub
#             r = randn()
#         end
#         return r

#     else 
#         span = ub - lb
#         if lb > 0 && span > 2.0 / (lb + sqrt(lb^2 + 4.0)) * exp((lb^2 - lb * sqrt(lb^2 + 4.0)) / 4.0)
#             a = (lb + sqrt(lb^2 + 4.0))/2.0
#             while true
#                 r = rand(Exponential(1.0 / a)) + lb
#                 u = rand()
#                 if u < exp(-0.5 * (r - a)^2) && r < ub
#                     return r
#                 end
#             end
#         elseif ub < 0 && ub - lb > 2.0 / (-ub + sqrt(ub^2 + 4.0)) * exp((ub^2 + ub * sqrt(ub^2 + 4.0)) / 4.0)
#             a = (-ub + sqrt(ub^2 + 4.0)) / 2.0
#             while true
#                 r = rand(Exponential(1.0 / a)) - ub
#                 u = rand()
#                 if u < exp(-0.5 * (r - a)^2) && r < -lb
#                     return -r
#                 end
#             end
#         else 
#             while true
#                 r = lb + rand() * (ub - lb)
#                 u = rand()
#                 if lb > 0
#                     rho = exp((lb^2 - r^2) * 0.5)
#                 elseif ub < 0
#                     rho = exp((ub^2 - r^2) * 0.5)
#                 else
#                     rho = exp(-r^2 * 0.5)
#                 end
#                 if u < rho
#                     return r
#                 end
#             end
#         end
#     end
# end

