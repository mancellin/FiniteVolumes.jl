using StaticArrays
import Base.transpose
import Base.rot180
import Base.rotl90
import Base.rotr90

"""
    Stencil{N, M, T}

A 2D stencil of horitontal width N and vertical height M, containing data of type T.

Indices follow the convention of cartesian indices, and NOT row-column as for matrices.

Basically a StaticOffsetArray with some custom methods.
"""
struct Stencil{N, M, T}
    data::SMatrix{N, M, T}
end

# Initialization
Stencil(s::AbstractVector) = Stencil{length(s), 1, eltype(s)}(s)
Stencil(s::AbstractMatrix) = Stencil(SMatrix{size(s)..., eltype(s)}(s...))

# Basic data access
Base.size(::Type{Stencil{N, M, T}}) where {N, M, T} = (N, M)
Base.size(s::Stencil) = size(typeof(s))
Base.eltype(s::Stencil{N, M, T}) where {N, M, T} = T

_offset_i(::Stencil{N, M}) where {N, M} = (N - 1) ÷ 2 + 1
_offset_j(::Stencil{N, M}) where {N, M} = (M - 1) ÷ 2 + 1

Base.getindex(s::Stencil, i::Int) = getindex(s.data, i + _offset_i(s), 1)
Base.getindex(s::Stencil, i::Int, j::Int) = getindex(s.data, i + _offset_i(s), j + _offset_j(s))

stencil_radiuses(N, M) = ((M-1) ÷ 2, (N-1) ÷ 2)
stencil_radiuses(::Stencil{N, M, T}) where {N, M, T} = stencil_radiuses(N, M)

Base.print(io::IO, st::Stencil) = print(io, "Stencil $(st.data')")
Base.show(io::IO, st::Stencil) = print(io, "Stencil $(st.data')")

# Transformation
Base.map(f, s::Stencil{N, M}) where {N, M} = Stencil(map(f, s.data))
Base.getindex(a::Vector{T}, s::Stencil{N, M, Int}) where {N, M, T} = Stencil{N, M, T}(a[s.data])

@generated function Base.transpose(s::Stencil{N, M, T}) where {N, M, T}
    indices = permutedims([(i, j) for i in 1:N, j in 1:M], (2, 1))
    items = [:(s.data[$i, $j]) for (i, j) in indices]
    quote
        Stencil{$M, $N, $T}(SMatrix{$M, $N, $T}($(items...)))
    end
end

@generated function upsidedown(s::Stencil{N, M, T}) where {N, M, T}
    indices = [(i, M-j+1) for i in 1:N, j in 1:M]
    items = [:(s.data[$i, $j]) for (i, j) in indices]
    quote
        Stencil{$N, $M, $T}(SMatrix{$N, $M, $T}($(items...)))
    end
end

@generated function more_above(s::Stencil{N, M, T}) where {N, M, T}
    below = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if j < (M+1)/2]
    above = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if j > (M+1)/2]
    quote
        +($(above...)) > +($(below...))
    end
end

@generated function more_below(s::Stencil{N, M, T}) where {N, M, T}
    below = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if j < (M+1)/2]
    above = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if j > (M+1)/2]
    quote
        +($(above...)) < +($(below...))
    end
end

@generated function rightsideleft(s::Stencil{N, M, T}) where {N, M, T}
    indices = [(N-i+1, j) for i in 1:N, j in 1:M]
    items = [:(s.data[$i, $j]) for (i, j) in indices]
    quote
        Stencil{$N, $M, $T}(SMatrix{$N, $M, $T}($(items...)))
    end
end

@generated function more_on_the_right(s::Stencil{N, M, T}) where {N, M, T}
    right = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if i > (N+1)/2]
    left = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if i < (N+1)/2]
    quote
        +($(left...)) < +($(right...))
    end
end

@generated function more_on_the_left(s::Stencil{N, M, T}) where {N, M, T}
    right = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if i > (N+1)/2]
    left = [:(s.data[$i, $j]) for i in 1:N, j in 1:M if i < (N+1)/2]
    quote
        +($(left...)) > +($(right...))
    end
end

for f in [:rotr90, :rotl90, :rot180]
    eval(
         quote
             @generated function Base.$f(s::Stencil{N, M, T}) where {N, M, T}
                 # Defines the function $f on Stencil.
                 # At compile time, call the function from Base on an array of indices
                 # and use the indices to compile a function on Stencils that does not call the function from Base.
                 indices = $f([(i, j) for i in 1:N, j in 1:M])
                 items = [:(s.data[$i, $j]) for (i, j) in indices]
                 new_N, new_M = size(indices)
                 quote
                     Stencil{$new_N, $new_M, $T}(SMatrix{$new_N, $new_M, $T}($(items...)))
                 end
             end
         end
        )
end

more_than_half_full(s::Stencil) = sum(s.data)/length(s.data) > 0.5
more_than_half_empty(s::Stencil) = sum(s.data)/length(s.data) < 0.5
invert(s::Stencil) =  typeof(s)(1.0 .- s.data)


###################
#  RegularMesh1D  #
###################

# TODO: implement N×1 stencil on RegularMesh1D for all N
function Stencil{3, 1}(grid::RegularMesh1D, i_cell)
    left_cell = i_cell == 1 ? 1 : i_cell - 1
    right_cell = i_cell == nb_cells(grid) ? nb_cells(grid) : i_cell + 1
    return Stencil(SVector{3, Int}(left_cell, i_cell, right_cell))
end

function Stencil{N, 1}(mesh::RegularMesh1D, i_cell, i_face) where N
    st = Stencil{N, 1}(mesh, i_cell)
    if face_center_relative_to_cell(mesh, i_cell, i_face)' * rotation_matrix(mesh, i_face)[:, 1] > 0
        return st
    else
        return rightsideleft(st)
    end
end


###################
#  RegularMesh2D  #
###################

function _right_cell(grid::AbstractRegularMesh2D, i_cell)
	if i_cell % grid.nx == 0  # Last cell at the end of a row
        if grid isa PeriodicRegularMesh2D
            return i_cell + 1 - grid.nx
        else
            return i_cell  # "Neumann" stencil
        end
	else  # General case
		return i_cell + 1
	end
end

function _left_cell(grid::AbstractRegularMesh2D, i_cell)
	if i_cell % grid.nx == 1  # First cell at the beggining of a row
        if grid isa PeriodicRegularMesh2D
            return i_cell - 1 + grid.nx
        else
            return i_cell
        end
	else
		return i_cell - 1
	end
end

_above_cell(grid::RegularMesh2D, i_cell) = i_cell + grid.nx <= nb_cells(grid) ? i_cell + grid.nx : i_cell
_below_cell(grid::RegularMesh2D, i_cell) = i_cell - grid.nx >= 1 ? i_cell - grid.nx : i_cell

_above_cell(grid::PeriodicRegularMesh2D, i_cell) = mod(i_cell+grid.nx, Base.OneTo(nb_cells(grid)))
_below_cell(grid::PeriodicRegularMesh2D, i_cell) = mod(i_cell-grid.nx, Base.OneTo(nb_cells(grid)))

function Stencil{3, 1}(grid::AbstractRegularMesh2D, i_cell)
    stencil = @SMatrix [_left_cell(grid, i_cell) i_cell _right_cell(grid, i_cell)]
    return Stencil{3, 1, Int}(stencil)
end

function Stencil{1, 3}(grid::AbstractRegularMesh2D, i_cell)
    stencil = @SMatrix [_below_cell(grid, i_cell); i_cell; _above_cell(grid, i_cell)]
    return Stencil{1, 3, Int}(stencil)
end

function Stencil{3, 3}(grid::AbstractRegularMesh2D, i_cell)
    left = _left_cell(grid, i_cell)
    right = _right_cell(grid, i_cell)
    stencil = @SMatrix [_below_cell(grid, left)   left    _above_cell(grid, left);
                        _below_cell(grid, i_cell) i_cell  _above_cell(grid, i_cell);
                        _below_cell(grid, right)  right   _above_cell(grid, right)]
    return Stencil{3, 3, Int}(stencil)
end

function Stencil{5, 5}(grid::AbstractRegularMesh2D, i_cell)
    left = _left_cell(grid, i_cell)
    right = _right_cell(grid, i_cell)
    leftleft = _left_cell(grid, left)
    rightright = _right_cell(grid, right)
    stencil = @SMatrix [_below_cell(grid, _below_cell(grid, leftleft))   _below_cell(grid, leftleft)   leftleft   _above_cell(grid, leftleft)   _above_cell(grid, _above_cell(grid, leftleft));
                        _below_cell(grid, _below_cell(grid, left))       _below_cell(grid, left)       left       _above_cell(grid, left)       _above_cell(grid, _above_cell(grid, left));
                        _below_cell(grid, _below_cell(grid, i_cell))     _below_cell(grid, i_cell)     i_cell     _above_cell(grid, i_cell)     _above_cell(grid, _above_cell(grid, i_cell));
                        _below_cell(grid, _below_cell(grid, right))      _below_cell(grid, right)      right      _above_cell(grid, right)      _above_cell(grid, _above_cell(grid, right));
                        _below_cell(grid, _below_cell(grid, rightright)) _below_cell(grid, rightright) rightright _above_cell(grid, rightright) _above_cell(grid, _above_cell(grid, rightright))]
    return Stencil{5, 5, Int}(stencil)
end

"""Stencil centered in i_cell, such that i_face is on the right."""
function Stencil{N, M}(mesh::AbstractRegularMesh2D, i_cell, i_face) where {N, M}
    if _is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[1]
        return rotr90(Stencil{M, N}(mesh, i_cell))
        # Compute a M×N stencil before the rotation to have a N×M stencil afterwards
    elseif _is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[2]
        return rotl90(Stencil{M, N}(mesh, i_cell))
    elseif !_is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[2]
        return rot180(Stencil{N, M}(mesh, i_cell)) 
    else
        return Stencil{N, M}(mesh, i_cell)
    end
end


##########################
#  Unused at the moment  #
##########################

function central_differences_gradient(grid::AbstractRegularMesh2D, w, i_cell)
    st = Stencil{3, 3}(grid, i_cell)
	dwdx = (w[st[1, 0]] - w[st[-1, 0]])/(2dx(grid))
	dwdy = (w[st[0, 1]] - w[st[0, -1]])/(2dy(grid))
	return SVector{2, Float64}(dwdx, dwdy)
end

function youngs_gradient(grid::AbstractRegularMesh2D, w, i_cell)
    st = Stencil{3, 3}(grid, i_cell)
	dwdx = (w[st[1, 1]] + 2w[st[1, 0]] + w[st[1, -1]] - w[st[-1, 1]] - 2w[st[-1, 0]] - w[st[-1, -1]])/(8dx(grid))
	dwdy = (w[st[1, 1]] + 2w[st[0, 1]] + w[st[-1, 1]] - w[st[1, -1]] - 2w[st[0, -1]] - w[st[-1, -1]])/(8dy(grid))
	return SVector{2, Float64}(dwdx, dwdy)
end

function least_square_gradient(grid::AbstractRegularMesh2D, w, i_cell)
    st = Stencil{3, 3}(grid, i_cell)
    A = @SMatrix zeros(2, 2)
    b = @SVector zeros(2)
	for i in -1:1, j in -1:1
		Δx = i*dx(grid)
		Δy = j*dy(grid)
		Δu = w[st[i, j]] - w[st[0, 0]]
		A = A + SMatrix{2, 2, Float64}(Δx^2, Δx*Δy, Δx*Δy, Δy^2)
		b = b + SVector{2, Float64}(Δu*Δx, Δu*Δy)
	end

    if A[2, 2] == 0.0
        #= @warning "2D mesh that is actually 1D" =#
        return [b[1] / A[1, 1], 0.0]
    else
        return A \ b
    end
end

