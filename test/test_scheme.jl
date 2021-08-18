# Test of some schemes

using Test
using StaticArrays
using FiniteVolumes
using FiniteVolumes.CartesianMeshes: Half, dx

@testset "Schemes" begin

@testset "Upwind linear advection" begin
    scheme = Upwind()
    @test scheme isa FiniteVolumes.Scheme

    # 1D
    mesh = CartesianMesh(0.0, 1.0, 2)
    left_flux = LinearAdvectionFlux(1.0)
    right_flux = LinearAdvectionFlux(-1.0)

    middle_face = (Half(3),)
    @test scheme(left_flux, mesh, [1.0, 2.0], middle_face) == 1.0
    @test scheme(right_flux, mesh, [1.0, 2.0], middle_face) == -2.0

    # Upwind scheme should ignore the time step passed as 5th argument
    @test scheme(left_flux, mesh, [1.0, 2.0], middle_face, 0.2) == 1.0
    @test scheme(left_flux, mesh, [1.0, 2.0], middle_face, nothing) == 1.0

    # Other types from StaticArrays
    @test scheme(left_flux, mesh, [Scalar(1.0), Scalar(2.0)], middle_face) == Scalar(1.0)
    @test scheme(left_flux, mesh, [SVector(1.0), SVector(2.0)], middle_face) == SVector(1.0)
    @test scheme(left_flux, mesh, [SVector(1.0, 2.0), SVector(2.0, 2.0)], middle_face) == SVector(1.0, 2.0)
    @test scheme(right_flux, mesh, [Scalar(1.0), Scalar(2.0)], middle_face) == -Scalar(2.0)
    @test scheme(right_flux, mesh, [SVector(1.0), SVector(2.0)], middle_face) == -SVector(2.0)
    @test scheme(right_flux, mesh, [SVector(1.0, 2.0), SVector(2.0, 2.0)], middle_face) == -SVector(2.0, 2.0)

    # 2D horizontal
    hmesh_2d = PeriodicCartesianMesh(2, 1)
    left_flux_2d = LinearAdvectionFlux([1.0, 0.0])
    right_flux_2d = LinearAdvectionFlux([-1.0, 0.0])

    middle_face = (Half(3), Half(2))
    @test scheme(left_flux_2d, hmesh_2d, [1.0, 2.0], middle_face) == 1.0
    @test scheme(left_flux_2d, hmesh_2d, [SVector(1.0), SVector(2.0)], middle_face) == SVector(1.0)
    @test scheme(left_flux_2d, hmesh_2d, [SVector(1.0, 1.0), SVector(2.0, 2.0)], middle_face) == SVector(1.0, 1.0)
    @test scheme(right_flux_2d, hmesh_2d, [1.0, 2.0], middle_face) == -2.0
    @test scheme(right_flux_2d, hmesh_2d, [SVector(1.0), SVector(2.0)], middle_face) == -SVector(2.0)
    @test scheme(right_flux_2d, hmesh_2d, [SVector(1.0, 1.0), SVector(2.0, 2.0)], middle_face) == -SVector(2.0, 2.0)

    # 2D vertical
    vmesh_2d = PeriodicCartesianMesh(1, 2)
    bottom_flux = LinearAdvectionFlux([0.0, 1.0])
    top_flux = LinearAdvectionFlux([0.0, -1.0])

    middle_face = (Half(2), Half(3))
    @test scheme(bottom_flux, vmesh_2d, [1.0 2.0], middle_face) == 1.0
    @test scheme(top_flux, vmesh_2d, [1.0 2.0], middle_face) == -2.0
end

@testset "Courant" begin
    mesh = CartesianMesh(2)
    left_flux = LinearAdvectionFlux(1.0)
    faster_left_flux = LinearAdvectionFlux(10.0)
    right_flux = LinearAdvectionFlux(-1.0)

    @test FiniteVolumes.courant(1.0, left_flux, mesh, [1.0, 1.0]) == 2.0
    @test FiniteVolumes.courant(1.0, faster_left_flux, mesh, [1.0, 1.0]) == 20.0
    @test FiniteVolumes.courant(1.0, right_flux, mesh, [1.0, 1.0]) == 2.0
end

@testset "Div" begin
    # 1D
    flux = LinearAdvectionFlux(2.0)
    mesh = PeriodicCartesianMesh(45)
    dt = 0.4*dx(mesh)[1]
    w0 = rand(FiniteVolumes.nb_cells(mesh))
    D = FiniteVolumes.inner_faces_to_cells_matrix(mesh)
    Δw1 = D * FiniteVolumes.numerical_fluxes(flux, mesh, w0, Upwind(), dt)
    Δw2 = -FiniteVolumes.div(flux, mesh, w0, Upwind(), dt)
    @test Δw1 ≈ Δw2

    # 2D
    flux = LinearAdvectionFlux([9.0, 4.0])
    mesh = PeriodicCartesianMesh(5, 8)
    dt = 0.4*dx(mesh)[1]
    w0 = rand(size(FiniteVolumes.cell_centers(mesh))...)
    D = FiniteVolumes.inner_faces_to_cells_matrix(mesh)
    Δw1 = D * FiniteVolumes.numerical_fluxes(flux, mesh, w0, Upwind(), dt)
    Δw2 = -FiniteVolumes.div(flux, mesh, w0, Upwind(), dt)
    @test reshape(Δw1, Tuple(mesh.nb_cells)) ≈ Δw2
end

end
