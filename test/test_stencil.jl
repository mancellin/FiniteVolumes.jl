# Test of the Stencil objects

using Test
using FiniteVolumes
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

        @test FiniteVolumes.upsidedown(st)[0, 1] == st[0, -1]
        @test FiniteVolumes.upsidedown(st)[1, 1] == st[1, -1]

        @test FiniteVolumes.rightsideleft(st)[1, 0] == st[-1, 0]
        @test FiniteVolumes.rightsideleft(st)[1, 1] == st[-1, 1]

        @test FiniteVolumes.more_above(st)
        @test FiniteVolumes.more_below(FiniteVolumes.upsidedown(st))

        @test FiniteVolumes.more_on_the_right(st)
        @test FiniteVolumes.more_on_the_left(FiniteVolumes.rightsideleft(st))

        st = Stencil(reshape(collect(1:3), 3, 1))
        @test size(transpose(st)) == (1, 3)
        @test FiniteVolumes.upsidedown(st) == st

        @test map(x -> 2x, st) |> typeof == Stencil{3, 1, Int}
        @test map(Float64, st) |> typeof == Stencil{3, 1, Float64}
    end

    @testset "From mesh" begin
        m = RegularMesh2D(5, 5)
        @test Stencil{3, 3}(m, 1).data == @SMatrix [1 1 6; 1 1 6; 2 2 7]
        @test Stencil{3, 3}(m, 12).data == @SMatrix [6 11 16; 7 12 17; 8 13 18]

        m = RegularMesh2D(5, 2)  # Two rows of 5 cells
        @test Stencil{3, 3}(m, 1).data == @SMatrix [1 1 6; 1 1 6; 2 2 7]

        grid = PeriodicRegularMesh2D(3, 3)
        st = Stencil{3, 3}(grid, 5)
        @test st[-1, -1] == 1
        @test st[1, 0] == 6
        @test st[1, 1] == 9

        st = Stencil{3, 3}(grid, 1)
        @test st[-1, -1] == 9
        @test st[1, 0] == 2
        @test st[1, 1] == 5

        st = Stencil{3, 3}(grid, 3)
        @test st[-1, -1] == 8
        @test st[1, 0] == 1
        @test st[1, 1] == 4

        st = Stencil{5, 5}(grid, 5)
        @test st[-1, -1] == 1
        @test st[-2, -1] == 3
        @test st[-2, -2] == 9
        @test st[1, 1] == 9
        @test st[2, 2] == 1
    end

    @testset "Gradients" begin
        grid = PeriodicRegularMesh2D(3, 3)

        methods = [
                   FiniteVolumes.central_differences_gradient,
                   FiniteVolumes.youngs_gradient,
                   FiniteVolumes.least_square_gradient,
                  ]

        horizontal = Float64[0, 0, 0, 1/2, 1/2, 1/2, 1, 1, 1]
        for f in methods
            g = f(grid, horizontal, 5)
            @test all(g .≈ [0.0, 3/2])
        end

        diagonal = Float64[-1/2, 0, 1/2, 0, 1/2, 1, 1/2, 1, 3/2]
        for f in methods
            g = f(grid, diagonal, 5)
            @test all(g .≈ [3/2, 3/2])
        end

        interface_1 = [1, 1, 1, 1/6, 1/2, 5/6, 0, 0, 0]
        g = FiniteVolumes.youngs_gradient(grid, interface_1, 5)
        @test -g[1]/g[2] ≈ 1/3

        #= g = FiniteVolumes.youngs_gradient(grid, [1, 1, 1, 1/3, 11/12, 1, 0, 1/12, 2/3], 5) =#
        #= @test -g[1]/g[2] ≈ 2/3 =#
    end
end
