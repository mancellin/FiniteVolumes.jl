using Test
using FiniteVolumes
using StaticArrays

@testset "Stencils" begin

    @testset "Initialization" begin
        # Explicit creation
        st = Stencil{3, 3, Float64}( @SMatrix zeros(3, 3) )
        @test st[0, 0] == 0.0
        @test size(st) == (3, 3)

        # Implicit creation
        # 3×3
        st = Stencil(reshape(collect(1:9), 3, 3))
        @test st[0, 0] == 5

        # 3×1
        st = Stencil(reshape(collect(1:3), 3, 1))
        @test st[-1, 0] == 1
        @test st[0, 0] == 2
        @test st[1, 0] == 3

        # 5×5
        st = Stencil(reshape(collect(1:25), 5, 5))
        @test st[0, 0] == 13
        @test st[2, 2] == 25

        st = Stencil(reshape(SVector{1, Float64}.(collect(1:9)), 3, 3))
        @test st[0, 0] |> typeof == SVector{1, Float64}
    end

    @testset "Basic transformation" begin
        st = Stencil(reshape(collect(1:9), 3, 3))
        @test transpose(st)[1, 1] == st[1, 1]
        @test transpose(st)[0, 1] == st[1, 0]

        @test rot180(st)[1, 1] == st[-1, -1]
        @test rot180(st)[0, 1] == st[0, -1]
    end
end
