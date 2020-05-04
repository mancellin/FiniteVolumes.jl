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
    flux = Muscl(limiter=minmod, flag=all_cells)
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
    flux = Muscl(limiter=minmod, flag=no_cell)
    @test flux(grid, from_left, w, [], 3) == Upwind()(grid, from_left, w, [[], [], []], 3)
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
