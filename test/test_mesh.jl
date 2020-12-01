# Tests of the AbstractMesh objects in FiniteVolumes.jl

using Test
using StaticArrays
using FiniteVolumes

@testset "2D meshes" begin
    @testset "Square 2D mesh" begin
        m = RegularMesh2D(5, 5)
        @test FiniteVolumes.nb_faces(m) == 60

        @test FiniteVolumes._is_horizontal(2)

        @test FiniteVolumes._is_on_right_edge(m, 9)
        @test !FiniteVolumes._is_on_right_edge(m, 14)
        @test FiniteVolumes._is_on_right_edge(m, 49)
        @test !FiniteVolumes._is_on_right_edge(m, 59)

        @test !FiniteVolumes._is_on_top_edge(m, 32)
        @test FiniteVolumes._is_on_top_edge(m, 42)
        @test !FiniteVolumes._is_on_top_edge(m, 49)
        @test FiniteVolumes._is_on_top_edge(m, 50)

        @test !FiniteVolumes._is_on_left_edge(m, 49)
        @test FiniteVolumes._is_on_left_edge(m, 51)
        @test !FiniteVolumes._is_on_left_edge(m, 56)
        @test FiniteVolumes._is_on_left_edge(m, 59)

        @test !FiniteVolumes._is_on_bottom_edge(m, 45)
        @test FiniteVolumes._is_on_bottom_edge(m, 52)
        @test !FiniteVolumes._is_on_bottom_edge(m, 55)
        @test FiniteVolumes._is_on_bottom_edge(m, 60)

        @test FiniteVolumes._is_inner_face(m, 41)
        @test !FiniteVolumes._is_inner_face(m, 42)

        @test FiniteVolumes.inner_faces(m) |> collect |> length == 40
        @test FiniteVolumes.boundary_faces(m) |> collect |> length == 20

        @test FiniteVolumes.cells_next_to_inner_face(m, 1) == (1, 2)
        @test FiniteVolumes.cells_next_to_inner_face(m, 10) == (5, 10)

        @test FiniteVolumes.cell_next_to_boundary_face(m, 19) == 10
        @test FiniteVolumes.cell_next_to_boundary_face(m, 42) == 21
        @test FiniteVolumes.cell_next_to_boundary_face(m, 51) == 1
        @test FiniteVolumes.cell_next_to_boundary_face(m, 54) == 2
        @test FiniteVolumes.cell_next_to_boundary_face(m, 57) == 16
        @test FiniteVolumes.cell_next_to_boundary_face(m, 60) == 5

        @test FiniteVolumes.cell_volume(m, 1) ≈ 0.04

        @test FiniteVolumes.cell_corners(m, 1).bottom_left == @SVector [0.0, 0.0]
        @test FiniteVolumes.cell_corners(m, nb_cells(m)).top_right == @SVector [1.0, 1.0]

        @test Stencil(m, 1).data == @SMatrix [1 1 6; 1 1 6; 2 2 7]
        @test Stencil(m, 12).data == @SMatrix [6 11 16; 7 12 17; 8 13 18]
    end

    @testset "Rectangular 2D mesh" begin
        m = RegularMesh2D(5, 2)  # Two rows of 5 cells
        @test FiniteVolumes.nb_faces(m) == 27

        @test FiniteVolumes._is_on_right_edge(m, 9)
        @test !FiniteVolumes._is_on_right_edge(m, 14)
        @test FiniteVolumes._is_on_right_edge(m, 19)
        @test !FiniteVolumes._is_on_right_edge(m, 29)

        @test FiniteVolumes._is_on_top_edge(m, 12)
        @test !FiniteVolumes._is_on_top_edge(m, 19)
        @test FiniteVolumes._is_on_top_edge(m, 20)

        @test FiniteVolumes._is_inner_face(m, 11)
        @test !FiniteVolumes._is_inner_face(m, 12)

        @test FiniteVolumes.inner_faces(m) |> collect |> length == 13
        @test FiniteVolumes.boundary_faces(m) |> collect |> length == 14

        @test FiniteVolumes.cell_next_to_boundary_face(m, 19) == 10
        @test FiniteVolumes.cell_next_to_boundary_face(m, 21) == 1

        @test FiniteVolumes.face_area(m, 1) == 0.5
        @test FiniteVolumes.face_area(m, 2) == 0.2
        
        @test FiniteVolumes.cell_volume(m, 1) == 0.1

        @test FiniteVolumes.cell_corners(m, 1).bottom_left == @SVector [0.0, 0.0]
        @test FiniteVolumes.cell_corners(m, nb_cells(m)).top_right == @SVector [1.0, 1.0]

        @test Stencil(m, 1).data == @SMatrix [1 1 6; 1 1 6; 2 2 7]
    end

    @testset "Stencils" begin

        grid = PeriodicRegularMesh2D(3, 3)
        st = Stencil(grid, 5)
        @test st[-1, -1] == 1
        @test st[1, 0] == 6
        @test st[1, 1] == 9

        st = Stencil(grid, 1)
        @test st[-1, -1] == 9
        @test st[1, 0] == 2
        @test st[1, 1] == 5

        st = Stencil(grid, 3)
        @test st[-1, -1] == 8
        @test st[1, 0] == 1
        @test st[1, 1] == 4

        stencil_radius = 2
        st = Stencil(grid, 5, stencil_radius)
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
