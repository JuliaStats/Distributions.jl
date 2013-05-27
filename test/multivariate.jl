n_samples = 100_000

for d in [MultivariateNormal([0.0, 0.0], [1.0 0.9; 0.9 1.0]),
          Dirichlet([2.0, 1.0]),
          Multinomial(1, [0.5, 0.5])]
    # Test rand
    s = rand(d)
    s = rand(d, 10)
    A = Array(eltype(s), 2, n_samples)
    rand!(d, A)

    # Test mean under CLT
    mu_hat = mean(A, 2)
    @assert norm(mu_hat - mean(d), Inf) < 0.1

    # Test cov under CLT
    sigma_hat = cov(A')
    @assert norm(sigma_hat - var(d), Inf) < 0.1

    # Test entropy and logpdf using AEP
    en_hat = -mean(logpdf(d, A))
    en = entropy(d)
    @assert abs(en_hat - en) < 0.1

    # Test entropy and pdf using AEP
    en_hat = -mean(log(pdf(d, A)))
    en = entropy(d)
    @assert abs(en_hat - en) < 0.1
end
