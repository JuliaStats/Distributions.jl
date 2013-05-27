@truncate Normal

function mean(d::TruncatedNormal)
    delta = pdf(d.untruncated, d.lower) - pdf(d.untruncated, d.upper)
    return mean(d.untruncated) + delta * var(d.untruncated) / d.nc
end

function modes(d::TruncatedNormal)
    mu = mean(d.untruncated)
    if d.upper < mu
        return [d.upper]
    elseif d.lower > mu
        return [d.lower]
    else
        return [mu]
    end
end

function var(d::TruncatedNormal) 
    s = std(d.untruncated)
    a = d.lower
    b = d.upper
    phi_a = pdf(d.untruncated, a) * s
    phi_b = pdf(d.untruncated, b) * s
    a_phi_a = a == -Inf ? 0.0 : a * phi_a
    b_phi_b = b == Inf ? 0.0 : b * phi_b
    z = d.nc
    return s^2 * (1 + (a_phi_a - b_phi_b) / z - ((phi_a - phi_b) / z)^2)
end

function entropy(d::TruncatedNormal)
    s = std(d.untruncated)
    a = d.lower
    b = d.upper
    phi_a = pdf(d.untruncated, a) * s
    phi_b = pdf(d.untruncated, b) * s
    a_phi_a = a == -Inf ? 0.0 : a * phi_a
    b_phi_b = b == Inf ? 0.0 : b * phi_b
    z = d.nc
    return entropy(d.untruncated) + log(z) +
           0.5 * (a_phi_a - b_phi_b) / z - 0.5 * ((phi_a - phi_b) / z)^2
end

# Rejection sampler based on algorithm from Robert (1992)
#  - Available at http://arxiv.org/abs/0907.4010
function randnt(lower::Real, upper::Real)
    if (lower <= 0 && upper == Inf) ||
       (upper >= 0 && lower == Inf) ||
       (lower <= 0 && upper >= 0 && upper - lower > sqrt(2.0 * pi))
        while true
            r = randn()
            if r > lower && r < upper
                return r
            end
        end
    elseif lower > 0 && upper - lower > 2.0 / (lower + sqrt(lower^2 + 4.0)) * exp((lower^2 - lower * sqrt(lower^2 + 4.0)) / 4.0)
        a = (lower + sqrt(lower^2 + 4.0))/2.0
        while true
            r = rand(Exponential(1.0 / a)) + lower
            u = rand()
            if u < exp(-0.5 * (r - a)^2) && r < upper
                return r
            end
        end    
    elseif upper < 0 && upper - lower > 2.0 / (-upper + sqrt(upper^2 + 4.0)) * exp((upper^2 + upper * sqrt(upper^2 + 4.0)) / 4.0)
        a = (-upper + sqrt(upper^2 + 4.0)) / 2.0
        while true
            r = rand(Exponential(1.0 / a)) - upper
            u = rand()
            if u < exp(-0.5 * (r - a)^2) && r < -lower
                return -r
            end
        end
    else
        while true
            r = lower + rand() * (upper - lower)
            u = rand()
            if lower > 0
                rho = exp((lower^2 - r^2) * 0.5)
            elseif upper < 0
                rho = exp((upper^2 - r^2) * 0.5)
            else
                rho = exp(-r^2 * 0.5)
            end
            if u < rho
                return r
            end
        end
    end
end

function rand(d::TruncatedNormal)
    mu = mean(d.untruncated)
    sigma = std(d.untruncated)
    z = randnt((d.lower - mu) / sigma, (d.upper - mu) / sigma)
    return mu + sigma * z
end
