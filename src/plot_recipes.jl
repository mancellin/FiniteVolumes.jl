using RecipesBase
using ColorTypes

@recipe function plot(grid::RegularMesh1D, w, var)
    seriestype := :steppre
    if var isa Symbol
        title --> var
    end

    x = [cell_center(grid, j)[1] + dx(grid)/2 for j in 1:nb_cells(grid)]
    data = [wi[var] for wi in w]
    x, data
end

@recipe function plot(grid::AbstractRegularMesh2D, w, var)
    seriestype := :heatmap
    seriescolor --> :viridis
    aspect_ratio --> 1
    if var isa Symbol
        title --> var
    end

    x = LinRange(grid.x_min, grid.x_max, grid.nx)
    y = LinRange(grid.y_min, grid.y_max, grid.ny)
    field = reshape([wi[var] for wi in w], (grid.nx, grid.ny))
    field = permutedims(field, (2, 1))
    x, y, field
end

@recipe function plot(grid::AbstractRegularMesh2D, w, vars::NTuple{N, Int};
                      base_colors=nothing) where N
    seriestype := :image
    yflip --> false

    if base_colors == nothing
        if N == 3
            base_colors = (XYZ(1.0, 0.0, 0.2), XYZ(0.0, 1.0, 0.2), XYZ(0.0, 0.0, 0.0))
        elseif N == 4
            base_colors = (XYZ(0.9, 0.0, 0.0), XYZ(0.0, 0.9, 0.0), XYZ(0.0, 0.0, 0.9), XYZ(0.0, 0.0, 0.0))
        end
    end
    base_colors = XYZ.(base_colors)

    x = LinRange(grid.x_min, grid.x_max, grid.nx)
    y = LinRange(grid.y_min, grid.y_max, grid.ny)
    pixels = [sum(wi[vars[k]]*base_colors[k] for k in 1:N) for wi in w]
    field = permutedims(reshape(pixels, (grid.nx, grid.ny)), (2, 1))
    x, y, field
end
