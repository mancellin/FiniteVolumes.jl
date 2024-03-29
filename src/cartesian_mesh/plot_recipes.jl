using RecipesBase
using ColorTypes

@recipe function plot(grid::AbstractCartesianMesh{1}, w, var=1)
    seriestype := :steppre
    if var isa Symbol
        title --> var
    end

    x = [cell_center(grid, j)[1] + dx(grid)[1]/2 for j in all_cells(grid)]
    data = [wi[var] for wi in w]
    x, data
end

@recipe function plot(grid::AbstractCartesianMesh{2}, w, var=1)
    seriestype := :heatmap
    seriescolor --> :viridis
    aspect_ratio --> 1
    if var isa Symbol
        title --> var
    end

    x = LinRange(grid.x_min[1], grid.x_max[1], grid.nb_cells[1])
    y = LinRange(grid.x_min[2], grid.x_max[2], grid.nb_cells[2])
    field = permutedims(map(wi -> wi[var], w), (2, 1))
    x, y, field
end

function get_base_colors(N)
    if N == 3
        return (RGB(0.643, 0.216, 0.255), RGB(0.666, 0.655, 0.224), RGB(0.176, 0.275, 0.443))
    elseif N == 4
        return (RGB(0.812, 0.408, 0.435), RGB(0.392, 0.69, 0.345), RGB(0.831, 0.757, 0.416), RGB(0.373, 0.322, 0.576))
    elseif N == 5
        return (RGB(0.812, 0.408, 0.435), RGB(0.392, 0.69, 0.345), RGB(0.831, 0.757, 0.416), RGB(0.373, 0.322, 0.576), RGB(0.2, 0.2, 0.2))
    end
end

@recipe function plot(grid::AbstractCartesianMesh{2}, w, vars::NTuple{N, Int};
                      base_colors=nothing) where N
    seriestype := :image
    yflip --> false

    if isnothing(base_colors)
        base_colors = get_base_colors(N)
    end
    base_colors = XYZ.(base_colors)

    x = LinRange(grid.x_min[1], grid.x_max[1], grid.nb_cells[1])
    y = LinRange(grid.x_min[2], grid.x_max[2], grid.nb_cells[2])
    pixels = map(wi -> sum(wi[vars[k]]*base_colors[k] for k in 1:N), w)
    field = permutedims(pixels, (2, 1))
    x, y, field
end
