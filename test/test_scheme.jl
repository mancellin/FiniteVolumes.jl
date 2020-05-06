using Test
using FiniteVolumes
using StaticArrays

@testset "Scalar upwind" begin
    flux = Upwind()

    @test flux isa NumericalFlux

    grid = RegularMesh1D(0.0, 1.0, 2)
    from_left = ScalarLinearAdvection(1.0)
    from_right = ScalarLinearAdvection(-1.0)

    @test flux(grid, from_left, [1.0, 0.0], [[], []], 2) == 1.0
    @test flux(grid, from_left, [0.0, 1.0], [[], []], 2) == 0.0
    @test flux(grid, from_right, [1.0, 0.0], [[], []], 2) == 0.0
    @test flux(grid, from_right, [0.0, 1.0], [[], []], 2) == -1.0

    @test flux(grid, from_left, [SVector(1.0), SVector(0.0)], [[], []], 2) == SVector(1.0)
    @test flux(grid, from_right, [SVector(1.0), SVector(0.0)], [[], []], 2) == SVector(0.0)

    hgrid_2d = PeriodicRegularMesh2D(0.0, 1.0, 2, 0.0, 1.0, 1)
    from_left_2d = ScalarLinearAdvection([1.0, 0.0])
    from_right_2d = ScalarLinearAdvection([-1.0, 0.0])

    @test flux(hgrid_2d, from_left_2d, [1.0, 0.0], [[], []], 1) == 1.0
    @test flux(hgrid_2d, from_right_2d, [1.0, 0.0], [[], []], 1) == 0.0
    @test flux(hgrid_2d, from_left_2d, [SVector(1.0), SVector(0.0)], [[], []], 1) == SVector(1.0)
    @test flux(hgrid_2d, from_right_2d, [SVector(1.0), SVector(0.0)], [[], []], 1) == SVector(0.0)

    from_left_2f_2d = ScalarLinearAdvection(2, [1.0, 0.0])
    from_right_2f_2d = ScalarLinearAdvection(2, [-1.0, 0.0])

    @test flux(hgrid_2d, from_left_2f_2d, [SVector(1.0, 1.0), SVector(0.0, 0.0)], [[], []], 1) == SVector(1.0, 1.0)
    @test flux(hgrid_2d, from_right_2f_2d, [SVector(1.0, 1.0), SVector(0.0, 0.0)], [[], []], 1) == SVector(0.0, 0.0)

    vgrid_2d = PeriodicRegularMesh2D(0.0, 1.0, 1, 0.0, 1.0, 2)
    from_bottom = ScalarLinearAdvection([0.0, 1.0])
    from_top = ScalarLinearAdvection([0.0, -1.0])

    @test flux(vgrid_2d, from_bottom, [1.0, 2.0], [[], []], 2) == 1.0
    @test flux(vgrid_2d, from_top, [1.0, 2.0], [[], []], 2) == -2.0
end

@testset "Courant" begin
    mesh = RegularMesh1D(0.0, 2.0, 2)
    from_left = ScalarLinearAdvection(1.0)
    faster_from_left = ScalarLinearAdvection(10.0)
    from_right = ScalarLinearAdvection(-1.0)

    @test FiniteVolumes.courant(1.0, mesh, from_left, [1.0, 1.0], [[], []]) == 1.0
    @test FiniteVolumes.courant(1.0, mesh, faster_from_left, [1.0, 1.0], [[], []]) == 10.0
    @test FiniteVolumes.courant(1.0, mesh, from_right, [1.0, 1.0], [[], []]) == 1.0
end

@testset "Scalar muscl" begin
    flux = Muscl(limiter=minmod)
    grid = RegularMesh1D(0.0, 1.0, 3)

    @test flux isa NumericalFlux

    from_left = ScalarLinearAdvection(1.0)
    @test flux(grid, from_left, [0.0, 0.5, 1.0], [], 3) == 0.75
    @test flux(grid, from_left, [0.0, 0.0, 1.0], [], 3) == 0.0
    @test flux(grid, from_left, [0.0, 1.0, 1.0], [], 3) == 1.0

    from_right = ScalarLinearAdvection(-1.0)
    @test flux(grid, from_right, [0.0, 0.5, 1.0], [], 2) == -0.25
    @test flux(grid, from_right, [0.0, 0.0, 1.0], [], 2) == 0.0
    @test flux(grid, from_right, [0.0, 1.0, 1.0], [], 2) == -1.0

    # Flagging no cells
    w = rand(3)
    flux = Either(no_cell, Muscl(limiter=minmod), Upwind())
    @test flux(grid, from_left, w, [[], [], []], 3) == Upwind()(grid, from_left, w, [[], [], []], 3)
end


@testset "Scalar Lagoutiere" begin
    flux = LagoutiereDownwind(β=0.2)
    grid = RegularMesh1D(0.0, 1.0, 3)

    @test flux isa NumericalFlux

    from_left = ScalarLinearAdvection(1.0)
    @test flux(grid, from_left, [0.0, 0.5, 1.0], [], 3) == 1.0
    @test flux(grid, from_left, [0.0, 0.0, 1.0], [], 3) == 0.0
    @test flux(grid, from_left, [0.0, 1.0, 1.0], [], 3) == 1.0

    from_right = ScalarLinearAdvection(-1.0)
    @test flux(grid, from_right, [0.0, 0.5, 1.0], [], 2) == 0.0
    @test flux(grid, from_right, [0.0, 0.0, 1.0], [], 2) == 0.0
    @test flux(grid, from_right, [0.0, 1.0, 1.0], [], 2) == -1.0

    w = rand(3)
    @test flux(grid, from_left, w, [], 3) ≈ Muscl(limiter=ultrabee(0.2))(grid, from_left, w, [], 3)
    @test flux(grid, from_right, w, [], 2) ≈ Muscl(limiter=ultrabee(0.2))(grid, from_right, w, [], 2)
end

@testset "VOF flux" begin
    grid = RegularMesh2D(3, 3)
    top_left = ScalarLinearAdvection([1.0, 1.0])
    bottom_right = ScalarLinearAdvection([-1.0, -1.0])
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wsupp = FiniteVolumes.compute_wsupp(top_left, w)

    upwind_flux = Upwind()
    upwind_vof = VOF(method=(α, β) -> α[0, 0], β=0.2)
    @test upwind_flux(grid, top_left, w, wsupp, 9) == upwind_vof(grid, top_left, w, wsupp, 9) == 0.5
    @test upwind_flux(grid, top_left, w, wsupp, 10) == upwind_vof(grid, top_left, w, wsupp, 10) == 0.5
    @test upwind_flux(grid, bottom_right, w, wsupp, 4) == upwind_vof(grid, bottom_right, w, wsupp, 4) == -0.5
    @test upwind_flux(grid, bottom_right, w, wsupp, 7) == upwind_vof(grid, bottom_right, w, wsupp, 7) == -0.5

    downwind_vof = VOF(method=(α, β) -> α[1, 0], β=0.2)
    @test downwind_vof(grid, top_left, w, wsupp, 4) == 0.5
    @test downwind_vof(grid, top_left, w, wsupp, 7) == 0.5
    @test downwind_vof(grid, top_left, w, wsupp, 9) == 0.6
    @test downwind_vof(grid, top_left, w, wsupp, 10) == 0.8 

    @test downwind_vof(grid, bottom_right, w, wsupp, 4) == -0.2
    @test downwind_vof(grid, bottom_right, w, wsupp, 7) == -0.4
    @test downwind_vof(grid, bottom_right, w, wsupp, 9) == -0.5
    @test downwind_vof(grid, bottom_right, w, wsupp, 10) == -0.5
end
