using Test
using FiniteVolumes
using StaticArrays

@testset "Scalar upwind" begin
    flux = Upwind()

    @test flux isa NumericalFlux

    grid = RegularMesh1D(0.0, 1.0, 2)
    from_left = ScalarLinearAdvection(1.0)
    from_right = ScalarLinearAdvection(-1.0)

    @test flux(from_left, grid, [1.0, 0.0], 2) == 1.0
    @test flux(from_left, grid, [0.0, 1.0], 2) == 0.0
    @test flux(from_right, grid, [1.0, 0.0], 2) == 0.0
    @test flux(from_right, grid, [0.0, 1.0], 2) == -1.0

    @test flux(from_left, grid, [Scalar(1.0), Scalar(0.0)], 2) == Scalar(1.0)
    @test flux(from_right, grid, [Scalar(1.0), Scalar(0.0)], 2) == Scalar(0.0)

    @test flux(from_left, grid, [SVector(1.0), SVector(0.0)], 2) == SVector(1.0)
    @test flux(from_right, grid, [SVector(1.0), SVector(0.0)], 2) == SVector(0.0)

    hgrid_2d = PeriodicRegularMesh2D(0.0, 1.0, 2, 0.0, 1.0, 1)
    from_left_2d = ScalarLinearAdvection([1.0, 0.0])
    from_right_2d = ScalarLinearAdvection([-1.0, 0.0])

    @test flux(from_left_2d, hgrid_2d, [1.0, 0.0], 1) == 1.0
    @test flux(from_right_2d, hgrid_2d, [1.0, 0.0], 1) == 0.0
    @test flux(from_left_2d, hgrid_2d, [SVector(1.0), SVector(0.0)], 1) == SVector(1.0)
    @test flux(from_right_2d, hgrid_2d, [SVector(1.0), SVector(0.0)], 1) == SVector(0.0)

    from_left_2f_2d = ScalarLinearAdvection(2, [1.0, 0.0])
    from_right_2f_2d = ScalarLinearAdvection(2, [-1.0, 0.0])

    @test flux(from_left_2f_2d, hgrid_2d, [SVector(1.0, 1.0), SVector(0.0, 0.0)], 1) == SVector(1.0, 1.0)
    @test flux(from_right_2f_2d, hgrid_2d, [SVector(1.0, 1.0), SVector(0.0, 0.0)], 1) == SVector(0.0, 0.0)

    vgrid_2d = PeriodicRegularMesh2D(0.0, 1.0, 1, 0.0, 1.0, 2)
    from_bottom = ScalarLinearAdvection([0.0, 1.0])
    from_top = ScalarLinearAdvection([0.0, -1.0])

    @test flux(from_bottom, vgrid_2d, [1.0, 2.0], 2) == 1.0
    @test flux(from_top, vgrid_2d, [1.0, 2.0], 2) == -2.0
end

@testset "Courant" begin
    mesh = RegularMesh1D(0.0, 2.0, 2)
    from_left = ScalarLinearAdvection(1.0)
    faster_from_left = ScalarLinearAdvection(10.0)
    from_right = ScalarLinearAdvection(-1.0)

    @test FiniteVolumes.courant(1.0, from_left, mesh, [1.0, 1.0]) == 1.0
    @test FiniteVolumes.courant(1.0, faster_from_left, mesh, [1.0, 1.0]) == 10.0
    @test FiniteVolumes.courant(1.0, from_right, mesh, [1.0, 1.0]) == 1.0
end

@testset "Scalar muscl" begin
    flux = Muscl(limiter=minmod)
    grid = RegularMesh1D(0.0, 1.0, 3)

    @test flux isa NumericalFlux

    from_left = ScalarLinearAdvection(1.0)
    @test flux(from_left, grid, [0.0, 0.5, 1.0], 3) == 0.75
    @test flux(from_left, grid, [0.0, 0.0, 1.0], 3) == 0.0
    @test flux(from_left, grid, [0.0, 1.0, 1.0], 3) == 1.0

    from_right = ScalarLinearAdvection(-1.0)
    @test flux(from_right, grid, [0.0, 0.5, 1.0], 2) == -0.25
    @test flux(from_right, grid, [0.0, 0.0, 1.0], 2) == 0.0
    @test flux(from_right, grid, [0.0, 1.0, 1.0], 2) == -1.0

    # Flagging no cells
    w = rand(3)
    flux = Hybrid(no_cell, Muscl(limiter=minmod), Upwind())
    @test flux(from_left, grid, w, 3) == Upwind()(from_left, grid, w, 3)
end


@testset "Scalar Lagoutiere" begin
    flux = LagoutiereDownwind()
    grid = RegularMesh1D(0.0, 3.0, 3)

    @test flux isa NumericalFlux

    from_left = ScalarLinearAdvection(1.0)
    @test flux(from_left, grid, [0.0, 0.5, 1.0], 3, dt=0.2) == 1.0
    @test flux(from_left, grid, [0.0, 0.0, 1.0], 3, dt=0.2) == 0.0
    @test flux(from_left, grid, [0.0, 1.0, 1.0], 3, dt=0.2) == 1.0

    from_right = ScalarLinearAdvection(-1.0)
    @test flux(from_right, grid, [0.0, 0.5, 1.0], 2, dt=0.2) == -0.5
    @test flux(from_right, grid, [0.0, 0.0, 1.0], 2, dt=0.2) == 0.0
    @test flux(from_right, grid, [0.0, 1.0, 1.0], 2, dt=0.2) == -1.0

    w = [0.0, rand(1)[1], 1.0]
    @test flux(from_left, grid, w, 3, dt=0.2) ≈ Muscl(limiter=ultrabee)(from_left, grid, w, 3, dt=0.2)
    @test flux(from_right, grid, w, 2, dt=0.2) ≈ Muscl(limiter=ultrabee)(from_right, grid, w, 2, dt=0.2)
end

@testset "VOF flux" begin
    grid = RegularMesh2D(3, 3)
    top_left = ScalarLinearAdvection([1.0, 1.0])
    bottom_right = ScalarLinearAdvection([-1.0, -1.0])
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    upwind_flux = Upwind()
    upwind_vof = VOF(method=(α, β) -> α[0, 0])
    @test upwind_flux(top_left, grid, w, 9) == upwind_vof(top_left, grid, w, 9) == 0.5
    @test upwind_flux(top_left, grid, w, 10) == upwind_vof(top_left, grid, w, 10) == 0.5
    @test upwind_flux(bottom_right, grid, w, 4) == upwind_vof(bottom_right, grid, w, 4) == -0.5
    @test upwind_flux(bottom_right, grid, w, 7) == upwind_vof(bottom_right, grid, w, 7) == -0.5

    downwind_vof = VOF(method=(α, β) -> α[1, 0])
    @test downwind_vof(top_left, grid, w, 4) == 0.5
    @test downwind_vof(top_left, grid, w, 7) == 0.5
    @test downwind_vof(top_left, grid, w, 9) == 0.6
    @test downwind_vof(top_left, grid, w, 10) == 0.8 

    @test downwind_vof(bottom_right, grid, w, 4) == -0.2
    @test downwind_vof(bottom_right, grid, w, 7) == -0.4
    @test downwind_vof(bottom_right, grid, w, 9) == -0.5
    @test downwind_vof(bottom_right, grid, w, 10) == -0.5
end
