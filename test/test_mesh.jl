using Test
using StaticArrays
using FiniteVolumes
using FiniteVolumes: Half

@testset "Mesh" begin
    @test FiniteVolumes.dx(CartesianMesh(10)) == SVector(0.1)
    @test FiniteVolumes.dx(CartesianMesh(10, 10)) == SVector(0.1, 0.1)
    @test FiniteVolumes.dx(CartesianMesh(SVector(1.0, 0.0), SVector(2.0, 2.0), SVector(10, 20))) == SVector(0.1, 0.1)

    @test FiniteVolumes.nb_cells(CartesianMesh(10)) == 10
    @test FiniteVolumes.nb_cells(CartesianMesh(5, 5)) == 25
    @test FiniteVolumes.nb_cells(CartesianMesh(2, 5)) == 10

    @test CartesianIndex(3, 3) in FiniteVolumes.all_cells(CartesianMesh(10, 10))
    @test FiniteVolumes.all_cells(CartesianMesh(2)) |> length == 2
    @test FiniteVolumes.all_cells(CartesianMesh(2, 2)) |> length == 4

    @test FiniteVolumes.inner_faces(CartesianMesh(2)) |> collect |> length == 1
    @test FiniteVolumes.inner_faces(PeriodicCartesianMesh(10, 10)) |> collect |> length == 200

    for m in [CartesianMesh(10), PeriodicCartesianMesh(3), CartesianMesh(5, 2), PeriodicCartesianMesh(10, 4)]
        @test FiniteVolumes.inner_faces(m) |> collect |> length == FiniteVolumes.nb_inner_faces(m)
        @test FiniteVolumes.boundary_faces(m) |> collect |> length == FiniteVolumes.nb_boundary_faces(m)
    end

    @test FiniteVolumes.boundary_faces(CartesianMesh(2)) |> collect |> length == 2
    # @test FiniteVolumes.boundary_faces(CartesianMesh(5, 5)) |> collect |> length == 20

    @test FiniteVolumes._direction((Half(3), Half(2))) == 1
    @test FiniteVolumes._direction((Half(2), Half(3))) == 2

    @test FiniteVolumes.cells_next_to_inner_face(PeriodicCartesianMesh(10), (Half(19),)) == (CartesianIndex(9), CartesianIndex(10))
    @test FiniteVolumes.cells_next_to_inner_face(PeriodicCartesianMesh(10), (Half(19),)) == (CartesianIndex(9), CartesianIndex(10))
    @test FiniteVolumes.cells_next_to_inner_face(PeriodicCartesianMesh(10), (Half(1),)) == (CartesianIndex(10), CartesianIndex(1))

    @test FiniteVolumes.cells_next_to_inner_face(CartesianMesh(10, 10), (Half(3), Half(2))) == (CartesianIndex(1, 1), CartesianIndex(2, 1))
    @test FiniteVolumes.cells_next_to_inner_face(CartesianMesh(10, 10), (Half(2), Half(3))) == (CartesianIndex(1, 1), CartesianIndex(1, 2))
    # @test FiniteVolumes.cell_next_to_boundary_face(CartesianMesh(10, 10), (Half(1), Half(2))) == CartesianIndex(1, 1)
    # @test FiniteVolumes.cell_next_to_boundary_face(CartesianMesh(10, 10), (Half(21), Half(2))) == CartesianIndex(10, 1)

    @test FiniteVolumes.cell_volume(CartesianMesh(10), CartesianIndex(3)) == 0.1
    @test FiniteVolumes.cell_volume(CartesianMesh(2, 5), CartesianIndex(1, 1)) == 0.1

    @test FiniteVolumes.face_area(CartesianMesh(5, 2), (Half(2), Half(1))) == 0.2
    @test FiniteVolumes.face_area(CartesianMesh(5, 2), (Half(1), Half(2))) == 0.5

    @test FiniteVolumes.cell_center(CartesianMesh(5, 5), CartesianIndex(1, 1)) ≈ [0.1, 0.1]
    @test FiniteVolumes.cell_center(CartesianMesh(5, 5), CartesianIndex(2, 1)) ≈ [0.3, 0.1]
    @test FiniteVolumes.face_center(CartesianMesh(5, 5), (Half(3), Half(2))) == [0.2, 0.1]
    @test FiniteVolumes.face_center(CartesianMesh(5, 5), (Half(2), Half(3))) == [0.1, 0.2]

    # @test FiniteVolumes.cell_corners(CartesianMesh(5, 5), ).bottom_left == @SVector [0.0, 0.0]
    # @test FiniteVolumes.cell_corners(CartesianMesh(5, 5), ).top_right == @SVector [1.0, 1.0]
end

# @btime cells_next_to_inner_face(mesh, i_face) setup=(mesh=CartesianMesh(10); i_face=(Half(19),))
# @btime cells_next_to_inner_face(mesh, i_face) setup=(mesh=PeriodicCartesianMesh(10); i_face=(Half(19),))
# @btime cell_center(mesh, (3,)) setup=(mesh=CartesianMesh(10))
# @btime face_center(mesh, (Half(3),)) setup=(mesh=CartesianMesh(10))

