#ifndef HYBRIDIR_AMPERE_HPP
#define HYBRIDIR_AMPERE_HPP

#include "vecfield.hpp"
#include "gridlayout.hpp"

#include <cstddef>
#include <iostream>

template<std::size_t dimension>
class Ampere
{
public:
    Ampere(std::shared_ptr<GridLayout<dimension>> grid)
        : m_grid{grid}
    {
        if (!m_grid)
            throw std::runtime_error("GridLayout is null");
    }

    void operator()(VecField<dimension> const& B, VecField<dimension>& J)
    {
        auto const dx = m_grid->cell_size(Direction::X);

        if constexpr (dimension == 1)
        {
            // TODO your code here

            // Jx = 0 everywhere on the grid (Dual)
            for (auto ix = m_grid->dual_dom_start(Direction::X);
                      ix <= m_grid->dual_dom_end(Direction::X); ++ix)
            J.x(ix) = 0.0;

            // Jy = -d(Bz)/dx, Jz = d(By)/dx, primal

            auto const start = m_grid->primal_dom_start(Direction::X);
            auto const end = m_grid->primal_dom_end(Direction::X);

            // define boundary value (can't use ix-1 at start)
            J.y(start) = 0.0;
            J.z(start) = 0.0;

            for (auto ix = start + 1;
                      ix <= end; ++ix)
            {
                // Backward difference using dual-centered B to compute primal-centered J 
                
                J.y(ix) = - (B.z(ix) - B.z(ix - 1)) / dx;
                J.z(ix) =  (B.y(ix) - B.y(ix - 1)) / dx;
            }
            
        }
        else
            throw std::runtime_error("Ampere not implemented for this dimension");
    }

private:
    std::shared_ptr<GridLayout<dimension>> m_grid;
};

#endif // HYBRIDIR_AMPERE_HPP
