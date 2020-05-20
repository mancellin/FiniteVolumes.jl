using Test
using FiniteVolumes
using FiniteVolumes: upsidedown, rightsideleft
using StaticArrays

@testset "Stencils" begin

    @testset "Initialization" begin
        # Explicit creation
        st = Stencil{3, 3, Float64}( @SMatrix rand(3, 3) )
        @test 0.0 <= st[0, 0] <= 1.0
        @test size(st) == (3, 3)

        st2 = Stencil(st.data)
        @test st[0, 0] == st2[0, 0]

        # Implicit creation
        # 3×3
        st = Stencil(reshape(collect(1:9), 3, 3))
        @test st[0, 0] == 5

        # 3×1
        st = Stencil(reshape(collect(1:3), 3, 1))
        @test st[-1, 0] == 1
        @test st[0, 0] == 2
        @test st[1, 0] == 3

        # 
        st = Stencil(SVector(1, 2, 3))
        @test typeof(st) == Stencil{3, 1, Int}

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

        @test upsidedown(st)[0, 1] == st[0, -1]
        @test upsidedown(st)[1, 1] == st[1, -1]

        @test rightsideleft(st)[1, 0] == st[-1, 0]
        @test rightsideleft(st)[1, 1] == st[-1, 1]

        @test FiniteVolumes.more_above(st)
        @test FiniteVolumes.more_below(upsidedown(st))

        @test FiniteVolumes.more_on_the_right(st)
        @test FiniteVolumes.more_on_the_left(rightsideleft(st))

        st = Stencil(reshape(collect(1:3), 3, 1))
        @test reverse(st)[-1, 0] == st[1, 0]
    end
end
