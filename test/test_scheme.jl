# Test of some schemes

using Test
using StaticArrays
using FiniteVolumes

@testset "Schemes" begin

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

end
