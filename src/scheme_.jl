using StaticArrays, ForwardDiff, LinearAlgebra
using Test
using BenchmarkTools

# MESH
cell_centers(mesh) = map(i -> cell_center(mesh, i), all_cells(mesh))

# Tools
struct Half{N} <: Real
    n::N
end
import Base.Int, Base.convert, Base.+, Base.print
Base.Int(h::Half{Int}) = Base.div(h.n, 2)
Base.convert(::Type{Int}, h::Half{Int}) = Int(h)
Base.convert(::Type{<:Number}, h::Half) = h.n/2
Base.:+(a::Half, b::Half) = Half(a.n+b.n)
Base.:-(a::Half, b::Half) = Half(a.n-b.n)
is_int(h::Half{Int}) = mod(h.n, 2) == 0
Base.print(io::IO, h::Half) = print(io, "$(h.n)/2")

abstract type AbstractCartesianMesh{D, L} end

struct CartesianMesh{D, L} <: AbstractCartesianMesh{D, L}
    x_min::SVector{D, L}
    x_max::SVector{D, L}
    nb_cells::SVector{D, Int64}
end
CartesianMesh(x_min::Number, x_max::Number, nb_cells::Int) = CartesianMesh{1, typeof(x_min)}(SVector(x_min), SVector(x_max), (nb_cells,))
CartesianMesh(nb_cells::Int) = CartesianMesh{1, Float64}(SVector(0.0), SVector(1.0), (nb_cells,))
CartesianMesh(nx::Int, ny::Int) = CartesianMesh{2, Float64}(SVector(0.0, 0.0), SVector(1.0, 1.0), (nx, ny))

struct PeriodicCartesianMesh{D, L} <: AbstractCartesianMesh{D, L}
    x_min::SVector{D, L}
    x_max::SVector{D, L}
    nb_cells::SVector{D, Int64}
end
PeriodicCartesianMesh(x_min::Number, x_max::Number, nb_cells::Int) = PeriodicCartesianMesh{1, typeof(x_min)}(SVector(x_min), SVector(x_max), (nb_cells,))
PeriodicCartesianMesh(nb_cells::Int) = PeriodicCartesianMesh{1, Float64}(SVector(0.0), SVector(1.0), (nb_cells,))
PeriodicCartesianMesh(nx::Int, ny::Int) = PeriodicCartesianMesh{2, Float64}(SVector(0.0, 0.0), SVector(1.0, 1.0), (nx, ny))

dx(mesh::AbstractCartesianMesh) = @. (mesh.x_max - mesh.x_min)/mesh.nb_cells

nb_cells(mesh::AbstractCartesianMesh) = prod(mesh.nb_cells)
all_cells(mesh::AbstractCartesianMesh) = CartesianIndices(Tuple(mesh.nb_cells))

inner_faces(mesh::CartesianMesh{1}) = ((Half(n),) for n in 3:2:2*mesh.nb_cells[1])
boundary_faces(mesh::CartesianMesh{1}) = ((Half(1),), (Half(2*mesh.nb_cells[1]+1),))

inner_faces(mesh::PeriodicCartesianMesh{1}) = ((Half(n),) for n in 1:2:2*mesh.nb_cells[1])
boundary_faces(mesh::PeriodicCartesianMesh{1}) = Tuple([]) 

function _face(dir, cell::CartesianIndex{2})
    if dir == 1
        Tuple(Half.(2*SVector((Tuple(cell)...))) - SVector(Half(1), Half(0)))
    elseif dir == 2
        Tuple(Half.(2*SVector((Tuple(cell)...))) - SVector(Half(0), Half(1)))
    else
        error()
    end
end
inner_faces(mesh::PeriodicCartesianMesh{2}) = (_face(dir, cell) for cell in CartesianIndices(Tuple(mesh.nb_cells)) for dir in 1:2)
boundary_faces(mesh::PeriodicCartesianMesh{2}) = Tuple([])

_direction(i_face::NTuple{2, Half{Int}}) = is_int(i_face[1]) ? 2 : 1

cells_next_to_inner_face(mesh::CartesianMesh{1}, i_face) = (CartesianIndex(Int(i_face[1])), CartesianIndex(Int(i_face[1]) + 1))
function cells_next_to_inner_face(mesh::PeriodicCartesianMesh{1}, i_face)
    if i_face == (Half(1),)
        (CartesianIndex(mesh.nb_cells[1]), CartesianIndex(1))
    else
        (CartesianIndex(Int(i_face[1])), CartesianIndex(Int(i_face[1]) + 1))
    end
end
function cells_next_to_inner_face(mesh::AbstractCartesianMesh{2}, i_face)
    if mesh isa PeriodicCartesianMesh
        if i_face[1] == Half(1)
            return (CartesianIndex(mesh.nb_cells[1], Int(i_face[2])), CartesianIndex(1, Int(i_face[2])))
        elseif i_face[2] == Half(1)
            return (CartesianIndex(Int(i_face[1]), mesh.nb_cells[2]), CartesianIndex(Int(i_face[1]), 1))
        end
    end
    if _direction(i_face) == 1
        return (CartesianIndex(Int(i_face[1]), Int(i_face[2])), CartesianIndex(Int(i_face[1]), Int(i_face[2])) + CartesianIndex(1, 0))
    elseif _direction(i_face) == 2
        return (CartesianIndex(Int(i_face[1]), Int(i_face[2])), CartesianIndex(Int(i_face[1]), Int(i_face[2])) + CartesianIndex(0, 1))
    else
        error()
    end
end

cell_next_to_boundary_face(mesh::CartesianMesh{1}, i_face) = i_face == (Half(1),) ? CartesianIndex(1) : CartesianIndex(mesh.nb_cells[1])

cell_center(mesh::AbstractCartesianMesh{1}, i_cell) = (i_cell[1] - 0.5) * dx(mesh)[1]
cell_center(mesh::AbstractCartesianMesh{N}, i_cell) where N = (SVector{N}(Tuple(i_cell)...) .- 0.5) .* dx(mesh)
face_center(mesh::AbstractCartesianMesh{1}, i_face) = (i_face[1] - 0.5) * dx(mesh)[1]
face_center(mesh::AbstractCartesianMesh{N}, i_face) where N = (SVector{N, Float64}(Tuple(i_face)...) .- 0.5) .* dx(mesh)


cell_volume(mesh::AbstractCartesianMesh, i_cell) = prod(dx(mesh))


face_area(mesh::AbstractCartesianMesh{1, T}, i_face) where T = one(T)
function face_area(mesh::AbstractCartesianMesh{2, T}, i_face) where T
    if _direction(i_face) == 1
        dx(mesh)[2]
    else
        dx(mesh)[1]
    end
end

normal_vector_type(::AbstractCartesianMesh{1, T}) where T = typeof(one(T))
normal_vector_type(::AbstractCartesianMesh{2, T}) where T = typeof(SVector(one(T), one(T)))

normal_vector(mesh::CartesianMesh{1, T}, i_face) where T = i_face == (Half(1),) ? -one(T) : one(T)
normal_vector(mesh::PeriodicCartesianMesh{1, T}, i_face) where T = one(T)
function normal_vector(mesh::AbstractCartesianMesh{2, T}, i_face) where T
    if mesh isa CartesianMesh
        if i_face[1] == Half(1)
            SVector(-oneunit(T), zero(T))./oneunit(T)
        elseif i_face[2] == Half(1)
            SVector(zero(T), -oneunit(T))./oneunit(T)
        end
    end
    if _direction(i_face) == 1
        SVector(oneunit(T), zero(T))./oneunit(T)
    elseif _direction(i_face) == 2
        SVector(zero(T), oneunit(T))./oneunit(T)
    else
        error()
    end
end


# FLUX

# Linear
struct LinearAdvectionFlux{V}
    velocity::V
end
LinearAdvectionFlux(velocity) = LinearAdvectionFlux{typeof(velocity)}(velocity)

LinearAlgebra.eigvals(f::LinearAdvectionFlux, w, n) = f.velocity'*n
(f::LinearAdvectionFlux)(w, n) = w*(f.velocity'*n)

# Wave equation
struct Wave1DFlux{T}
    velocity::T
end
Wave1DFlux(velocity) = Wave1DFlux{typeof(velocity)}(velocity)

(f::Wave1DFlux)(w, n) = f.velocity*SVector{2}(w[2], w[1])*n
jacobian(f::Wave1DFlux, w, n) = f.velocity*SMatrix{2, 2}(0.0, 1.0, 1.0, 0.0)*n
LinearAlgebra.eigen(f::Wave1DFlux, w, n) = SVector{2}(-1.0, 1.0)*n, SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n
# jacobian(f::Wave1DFlux, w::SVector{2, T}, n) where T = f.velocity*SMatrix{2, 2}(zero(T)/oneunit(T), one(T), one(T), zero(T)/oneunit(T))*n
# LinearAlgebra.eigen(f::Wave1DFlux, w::SVector{2, T}, n) where T = SVector{2}(-one(T), one(T))*n, SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n

# Generic
struct FluxFunction{T, D, F}
    func::F
end
FluxFunction{T, D}(f) where {T, D} = FluxFunction{T, D, typeof(f)}(f)

(f::FluxFunction{T, 1})(w, n) where T = f.func(w)*n
(f::FluxFunction)(w, n) = f.func(w)'*n
LinearAlgebra.eigvals(f::FluxFunction{<:Number}, w, n) = ForwardDiff.derivative(v -> f(v, n), w)

jacobian(f::FluxFunction, w, n) = ForwardDiff.jacobian(v -> f(v, n), w)
LinearAlgebra.eigvals(f::FluxFunction{T}, w, n) where {T <: AbstractArray} = eigen(jacobian(f, w, n)).values .|> real
function LinearAlgebra.eigen(f::FluxFunction{T}, w, n) where {T <: AbstractArray}
    λc, Lc = eigen(jacobian(f, w, n))
    return real.(λc), real.(Lc)
end


# SCHEME
abstract type NumericalFlux end

# By default, can ignore time step
(scheme::NumericalFlux)(model, mesh, w, i_face, dt) = scheme(model, mesh, w, i_face)

struct Upwind <: NumericalFlux end

# Single-wave problems
function (::Upwind)(flux::Union{LinearAdvectionFlux, FluxFunction{<:Number}}, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    λ = eigvals(flux, (w[i_cell_1] + w[i_cell_2])/2, n)
    return λ > zero(λ) ? flux(w[i_cell_1], n) : flux(w[i_cell_2], n)
end

# Multi-wave problems
function (::Upwind)(flux, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    w_mean = (w[i_cell_1] + w[i_cell_2])/2
    λ, L = eigen(flux, w_mean, n)
    left_to_right = λ .> zero(λ)
    if all(left_to_right)
        return flux(w[i_cell_1], n)
    elseif !any(left_to_right)
        return flux(w[i_cell_2], n)
    else
        f₁ = flux(w[i_cell_1], n)
        f₂ = flux(w[i_cell_2], n)
        return (f₁ + f₂ + L'*diagm(sign.(λ))*L*(f₁ - f₂))/2
    end
end

# BC

abstract type BoundaryCondition end

(scheme::BoundaryCondition)(model, mesh, w, i_face, dt) = scheme(model, mesh, w, i_face)

struct NeumannBC <: BoundaryCondition end

function (::NeumannBC)(flux, mesh, w, i_face)
    i_cell_1 = cell_next_to_boundary_face(mesh, i_face)
    return flux(w[i_cell_1], normal_vector(mesh, i_face))
end

# Div

function div!(Δw, flux, mesh, w, dt, numerical_flux, boundary_flux)
    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(flux, mesh, w, i_face, dt)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δw[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δw[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end
    @inbounds for i_face in boundary_faces(mesh)
        ϕ = boundary_flux(flux, mesh, w, i_face, dt)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δw[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
end

flux_type(f, m::AbstractCartesianMesh{N, L}, w::AbstractArray{W}) where {N, L, W} = typeof(f(w[1], normal_vector(m, iterate(inner_faces(m))[1]))/oneunit(L))

function div(flux, mesh, w; time_step=nothing, numerical_flux=Upwind(), boundary_flux=NeumannBC())
    Δw = zeros(flux_type(flux, mesh, w), size(w))
    div!(Δw, flux, mesh, w, time_step, numerical_flux, boundary_flux)
    return Δw
end

###########
#  tests  #
###########
using BenchmarkTools
using Test

@testset "Finite Volumes" begin
    @testset "Mesh" begin
        @test dx(CartesianMesh(10)) == SVector(0.1)
        @test dx(CartesianMesh(10, 10)) == SVector(0.1, 0.1)
        @test dx(CartesianMesh(SVector(1.0, 0.0), SVector(2.0, 2.0), SVector(10, 20))) == SVector(0.1, 0.1)
        @test nb_cells(CartesianMesh(10)) == 10
        @test nb_cells(CartesianMesh(5, 5)) == 25
        @test CartesianIndex(3, 3) in all_cells(CartesianMesh(10, 10))
        @test all_cells(CartesianMesh(2)) |> length == 2

        @test inner_faces(CartesianMesh(2)) |> collect |> length == 1
        @test boundary_faces(CartesianMesh(2)) |> collect |> length == 2
        @test inner_faces(PeriodicCartesianMesh(10, 10)) |> collect |> length == 200

        @test _direction((Half(3), Half(2))) == 1
        @test _direction((Half(2), Half(3))) == 2

        @test cells_next_to_inner_face(PeriodicCartesianMesh(10), (Half(19),)) == (CartesianIndex(9), CartesianIndex(10))
        @test cells_next_to_inner_face(PeriodicCartesianMesh(10), (Half(19),)) == (CartesianIndex(9), CartesianIndex(10))
        @test cells_next_to_inner_face(PeriodicCartesianMesh(10), (Half(1),)) == (CartesianIndex(10), CartesianIndex(1))
        @test cells_next_to_inner_face(CartesianMesh(10, 10), (Half(3), Half(2))) == (CartesianIndex(1, 1), CartesianIndex(2, 1))
        @test cells_next_to_inner_face(CartesianMesh(10, 10), (Half(2), Half(3))) == (CartesianIndex(1, 1), CartesianIndex(1, 2))

        # @btime cells_next_to_inner_face(mesh, i_face) setup=(mesh=CartesianMesh(10); i_face=(Half(19),))
        # @btime cells_next_to_inner_face(mesh, i_face) setup=(mesh=PeriodicCartesianMesh(10); i_face=(Half(19),))
        # @btime cell_center(mesh, (3,)) setup=(mesh=CartesianMesh(10))
        # @btime face_center(mesh, (Half(3),)) setup=(mesh=CartesianMesh(10))

        @test cell_volume(CartesianMesh(10), (3,)) == 0.1
    end

    @testset "Scalar 1D" begin
        @testset "Vanilla" begin
            mesh = CartesianMesh(10)
            w0 = map(x -> sin(2π*x), cell_centers(mesh))
            i_face = (Half(11),)

            flux = LinearAdvectionFlux(1.0)
            @test (Upwind())(flux, mesh, w0, i_face) == w0[5]
            # @btime (Upwind())($f, $mesh, $w, $i_face)  # ~4ns

            flux = LinearAdvectionFlux(-1.0)
            @test (Upwind())(flux, mesh, w0, i_face) == -w0[6]

            flux = FluxFunction{Float64, 1}(α -> α^2/2)
            @test (Upwind())(flux, mesh, w0, i_face) ≈ w0[5]^2/2
            # @btime (Upwind())($f, $mesh, $w, $i_face)  # ~4ns

            dt = 0.01
            flux = LinearAdvectionFlux(1.0)
            w = w0 .- dt*div(flux, mesh, w0)
            @test all(minimum(w0) .<= w .<= maximum(w0))
        end

        @testset "Measurements" begin
            using Measurements
            mesh = CartesianMesh(10)
            w0 = map(x -> sin(2π*x), cell_centers(mesh))
            i_face = (Half(11),)

            flux = LinearAdvectionFlux(1.0 ± 1.0)
            @test (Upwind())(flux, mesh, w0, i_face) |> Measurements.value == w0[5]

            dt = 0.01
            w = w0 .- dt*div(flux, mesh, w0)
        end

        @testset "Unitful" begin
            using Unitful: kg, m, s
            mesh = CartesianMesh(0.0m, 1.0m, 10)
            i_face = (Half(11),)
            dt = 0.01s

            flux = LinearAdvectionFlux(1m/s)
            w1 = map(x -> sin(2π*x/1m)*1kg/m, cell_centers(mesh))
            @test (Upwind())(flux, mesh, w1, i_face) |> typeof == typeof(w1[1]* 1m/s)
            w2 = w1 .- dt*div(flux, mesh, w1)
            @test eltype(w2) == eltype(w1)
        end
    end

    @testset "Multi-scalar 1D" begin
        @testset "Vanilla" begin
            mesh = CartesianMesh(10)
            w = map(x -> SVector{2}(sin(2π*x), cos(2π*x)), cell_centers(mesh))
            dt = 0.01
            i_face = (Half(11),)

            f = LinearAdvectionFlux(1.0)
            # @btime (Upwind())($f, $mesh, $w, $i_face, $dt)

            f = Wave1DFlux(1.0)
            # @btime (Upwind())($f, $mesh, $w, $i_face, $dt)

            f = FluxFunction{SVector{2}, 1}(α -> α.^2/2)
            # @btime (Upwind())($f, $mesh, $w, $i_face, $dt)

            # Wave equation
            f = FluxFunction{SVector{2}, 1}(α -> SVector(α[2], α[1]))
            # @btime (Upwind())($f, $mesh, $w, $i_face, $dt)
            (Upwind())(f, mesh, w, i_face)

            # Shallow waters
            f = FluxFunction{SVector{2}, 1}(v -> SVector(v[2], v[2]^2/v[1] + v[1]^2*9.81/2))
            (Upwind())(f, mesh, w, i_face, dt)
        end

        # @testset "Unitful" begin
        #     using Unitful: kg, m, s
        #     mesh = CartesianMesh(0.0m, 1.0m, 10)
        #     i_face = (Half(11),)
        #     dt = 0.01s
        #     flux = Wave1DFlux(1m/s)
        #     w1 = map(x -> (sin(2π*x/1m)*kg/m, 0kg/m/s), cell_centers(mesh))
        #     Δw = map(x -> (0kg/m/s, 0kg/m/s^2), cell_centers(mesh))
        #     w2 = w1 .- dt.*div(flux, mesh, w1)
        #     @test eltype(w2) == eltype(w1)
        # end

    end

    @testset "Scalar 2D" begin
        mesh = PeriodicCartesianMesh(10, 10)
        w = map(x -> sin(2π*x[1])*cos(2π*x[2]), cell_centers(mesh))

        flux = LinearAdvectionFlux(SVector(1.0, 0.0))
        i_face = (Half(11), Half(4))  # Vertical cell
        @test (Upwind())(flux, mesh, w, i_face) != 0.0
        i_face = (Half(4), Half(11))
        @test (Upwind())(flux, mesh, w, i_face) == 0.0

        div(flux, mesh, w)
    end

    @testset "Derivation" begin
        function run(v)
            f = LinearAdvectionFlux(v)
            mesh = CartesianMesh(0.0, 1.0, 100)
            w = [sin(2π*x) for x in cell_centers(mesh)]
            dt = 0.01
            w2 = w .- dt*div(f, mesh, w)
        end
        ForwardDiff.derivative(run, 1.0)
    end

end
