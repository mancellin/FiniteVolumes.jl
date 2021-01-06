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

        @test FiniteVolumes.cell_center(m, 1) ≈ [0.1, 0.1]
        @test FiniteVolumes.face_center(m, 1) == [0.2, 0.1]
        @test FiniteVolumes.cell_center(m, 2) ≈ [0.3, 0.1]
        @test FiniteVolumes.face_center(m, 2) == [0.1, 0.2]

        @test FiniteVolumes.face_center(m, 51) == [0.0, 0.1]
        @test FiniteVolumes.face_center(m, 52) == [0.1, 0.0]
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
    end
end
