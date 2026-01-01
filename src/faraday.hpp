#ifndef HYBRIDIR_FARADAY_HPP
#define HYBRIDIR_FARADAY_HPP

#include "vecfield.hpp"
#include "gridlayout.hpp"
#include "utils.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

template<std::size_t dimension>
class Faraday
{
    // TODO implement the Faraday class, hint - get inspiration from Ampere

public:
    Faraday(std::shared_ptr<GridLayout<dimension>> grid, double dt)
        : m_grid{grid}, m_dt{dt}
    {
        if (!m_grid)
            throw std::runtime_error("GridLayout is null");
    }

    void operator()(VecField<dimension> const& B, VecField<dimension> const& E, VecField<dimension>& Bnew)
    {
        auto const dx = m_grid->cell_size(Direction::X);

        if constexpr (dimension == 1)
        {
            // TODO your code here
            for (auto ix = m_grid->primal_dom_start(Direction::X);
                      ix <= m_grid->primal_dom_end(Direction::X); ++ix)
            {
                // Bx doesnt't change in 1D (curl_x = 0), primal
                Bnew.x(ix) = B.x(ix);

            }

                // Update By and Bz, dual 
            auto const dstart = m_grid->dual_dom_start(Direction::X);
            auto const dend   = m_grid->dual_dom_end(Direction::X);

            // boundary value (cannot access E.(dend+1) if we use forward diff)
            Bnew.y(dend) = B.y(dend);
            Bnew.z(dend) = B.z(dend);

            for (auto ix = dstart; ix < dend; ++ix)
        {
                // Faraday: dB/dt = -curl(E)
                // In 1D: By^{n+1} = By^n + dt * dEz/dx
                //        Bz^{n+1} = Bz^n - dt * dEy/dx

                Bnew.y(ix) = B.y(ix) + m_dt * (E.z(ix + 1) - E.z(ix)) / dx;
                Bnew.z(ix) = B.z(ix) - m_dt * (E.y(ix + 1) - E.y(ix)) / dx; 
            }
        }
        else
        {
            throw std::runtime_error("Faraday not implemented for this dimension");
        }
    }

private:
    std::shared_ptr<GridLayout<dimension>> m_grid;
    double m_dt;
};

#endif // HYBRIDIR_FARADAY_HPP
