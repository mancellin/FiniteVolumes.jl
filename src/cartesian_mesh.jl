# Tools
struct Half{N} <: Real
    n::N
end
import Base.Int, Base.convert, Base.+, Base.-, Base.<, Base.==, Base.print
Base.Int(h::Half{Int}) = Base.div(h.n, 2)
Base.convert(::Type{Int}, h::Half{Int}) = Int(h)
Base.convert(::Type{<:Number}, h::Half) = h.n/2
Base.:+(a::Half, b::Half) = Half(a.n+b.n)
Base.:-(a::Half) = Half(-a.n)
Base.:-(a::Half, b::Half) = Half(a.n-b.n)
Base.:(==)(a::Half, b::Half) = a.n == b.n
Base.:<(a::Half, b::Half) = a.n < b.n
is_int(h::Half{Int}) = mod(h.n, 2) == 0
Base.print(io::IO, h::Half) = print(io, "$(h.n)/2")


##################################
abstract type FiniteVolumeMesh end
##################################

cell_centers(mesh) = map(i -> cell_center(mesh, i), all_cells(mesh))

#############################################
abstract type AbstractCartesianMesh{D, L} end
#############################################

#########################################################
struct CartesianMesh{D, L} <: AbstractCartesianMesh{D, L}
    x_min::SVector{D, L}
    x_max::SVector{D, L}
    nb_cells::SVector{D, Int64}
end
CartesianMesh(x_min::Number, x_max::Number, nb_cells::Int) = CartesianMesh{1, typeof(x_min)}(SVector(x_min), SVector(x_max), (nb_cells,))
CartesianMesh(nb_cells::Int) = CartesianMesh{1, Float64}(SVector(0.0), SVector(1.0), (nb_cells,))
CartesianMesh(nx::Int, ny::Int) = CartesianMesh{2, Float64}(SVector(0.0, 0.0), SVector(1.0, 1.0), (nx, ny))

#################################################################
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

const _left = (-Half(1), Half(0))
const _right = (Half(1), Half(0))
const _down = (Half(0), -Half(1))
const _up = (Half(0), Half(1))
_bottom_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((1:n, 1:1)))
_top_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((1:n, m:m)))
_left_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((1:1, 1:m)))
_right_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((n:n, 1:m)))

_face(dir, cell::CartesianIndex{2}) = Half.(2 .* Tuple(cell)) .+ dir

inner_faces(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; (_face(dir, cell) for (dir, cells) in ((_right, CartesianIndices((1:(n-1), 1:m))), (_up, CartesianIndices((1:n, 1:(m-1))))) for cell in cells))
boundary_faces(mesh::CartesianMesh{2}) = (_face(dir, cell) for (dir, cells) in ((_down, _bottom_cells(mesh)), (_right, _right_cells(mesh)), (_up, _top_cells(mesh)), (_left, _left_cells(mesh))) for cell in cells)

inner_faces(mesh::PeriodicCartesianMesh{2}) = (_face(dir, cell) for cell in CartesianIndices(Tuple(mesh.nb_cells)) for dir in (_down, _left))
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

function cell_next_to_boundary_face(mesh::CartesianMesh{2}, i_face)
    if i_face[1] == Half(1)  # left boundary
        return CartesianIndex(Int(i_face[1]) + 1, Int(i_face[2]))
    elseif i_face[2] == Half(1)  # bottom boundary
        return CartesianIndex(Int(i_face[1]), Int(i_face[2]) + 1)
    elseif _direction(i_face) == 1  # right boundary
        return CartesianIndex(Int(i_face[1]), Int(i_face[2]))
    elseif _direction(i_face) == 2  # top boundary
        return CartesianIndex(Int(i_face[1]), Int(i_face[2]))
    else
        error()
    end
end

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

normal_vector(mesh::CartesianMesh{1, T}, i_face) where T = i_face == (Half(1),) ? -one(T) : one(T)
normal_vector(mesh::PeriodicCartesianMesh{1, T}, i_face) where T = one(T)
function normal_vector(mesh::AbstractCartesianMesh{2, T}, i_face) where T
    if mesh isa CartesianMesh
        if i_face[1] == Half(1)  # Left boundary
            return SVector(-oneunit(T), zero(T))./oneunit(T)
        elseif i_face[2] == Half(1)  # Bottom boundary
            return SVector(zero(T), -oneunit(T))./oneunit(T)
        end
    end
    if _direction(i_face) == 1
        return SVector(oneunit(T), zero(T))./oneunit(T)
    elseif _direction(i_face) == 2
        return SVector(zero(T), oneunit(T))./oneunit(T)
    else
        error()
    end
end

# function cell_corners(mesh::AbstractRegularMesh2D, i_cell)
#     c = cell_center(mesh, i_cell)
#     return (
#      bottom_left= SVector(c[1] - dx(mesh)/2, c[2] - dy(mesh)/2),
#      top_left=    SVector(c[1] - dx(mesh)/2, c[2] + dy(mesh)/2),
#      bottom_right=SVector(c[1] + dx(mesh)/2, c[2] - dy(mesh)/2),
#      top_right=   SVector(c[1] + dx(mesh)/2, c[2] + dy(mesh)/2),
#     )
# end


