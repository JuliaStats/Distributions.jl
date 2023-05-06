function wrapped_use_characteristic(d::Normal, l, u, tol)
    μ = d.μ
    σ = d.σ
    nterms_pdf = _num_terms_wrappednormal_pdf(μ, σ, tol)
    nterms_cf = _num_terms_wrappednormal_cf(μ, σ, tol)
    return nterms_cf < nterms_pdf
end

function _num_terms_wrappednormal_pdf(μ, σ, ϵ)
    T = Base.promote_eltype(μ, σ, ϵ)
    return 1 + ((σ / sqrt2) / π) * max(1, sqrt(-logtwo - log(ϵ) - (3//2) * T(logπ)))
end

function _num_terms_wrappednormal_cf(μ, σ, ϵ)
    T = Base.promote_eltype(μ, σ, ϵ)
    return max(1, sqrt(-T(logtwo) / 2 - logπ - log(σ) - log(ϵ))) / σ
end
