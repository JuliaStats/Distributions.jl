@testset "Constructor (#1471)" begin
    @test_throws DomainError InverseGamma(-1, 2)
    InverseGamma(-1, 2; check_args=false) # no error
end
