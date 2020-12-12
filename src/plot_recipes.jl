using RecipesBase
using ColorTypes

@recipe function plot(grid::RegularMesh1D, w, var)
    seriestype := :steppre
    if var isa Symbol
        title --> var
    end

    x = [cell_center(grid, j)[1] + dx(grid)/2 for j in all_cells(grid)]
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
            base_colors = (RGB(0.643, 0.216, 0.255), RGB(0.176, 0.275, 0.443), RGB(0.666, 0.655, 0.224))
        elseif N == 4
            base_colors = (RGB(0.812, 0.408, 0.435), RGB(0.831, 0.757, 0.416), RGB(0.373, 0.322, 0.576), RGB(0.392, 0.69, 0.345))
        end
    end
    base_colors = XYZ.(base_colors)

    x = LinRange(grid.x_min, grid.x_max, grid.nx)
    y = LinRange(grid.y_min, grid.y_max, grid.ny)
    pixels = [sum(wi[vars[k]]*base_colors[k] for k in 1:N) for wi in w]
    field = permutedims(reshape(pixels, (grid.nx, grid.ny)), (2, 1))
    x, y, field
end
