#ifndef PHARE_CORE_NUMERICS_PUSHER_PUSHER_HPP
#define PHARE_CORE_NUMERICS_PUSHER_PUSHER_HPP


#include "vecfield.hpp"
#include "particle.hpp"

#include <cstddef>
#include <vector>
#include <cmath>


template<std::size_t dimension>
class Pusher
{
protected:
    std::shared_ptr<GridLayout<dimension>> layout_;
    double dt_;

public:
    Pusher(std::shared_ptr<GridLayout<dimension>> layout, double dt)
        : layout_(layout)
        , dt_(dt)
    {
    }

    virtual void operator()(std::vector<Particle<dimension>>& particles,
                            VecField<dimension> const& E, VecField<dimension> const& B)
        = 0;

    virtual ~Pusher() {}
};



template<std::size_t dimension>
class Boris : public Pusher<dimension>
{
public:
    Boris(std::shared_ptr<GridLayout<dimension>> layout, double dt)
        : Pusher<dimension>{layout, dt}
    {
    }

    void operator()(std::vector<Particle<dimension>>& particles, VecField<dimension> const& E,
                    VecField<dimension> const& B) override
    {
        for (auto& particle : particles)
        {
            // TODO implement the Boris pusher

            double dt = this->dt_;

            // Step 1: Half-step position update
            particle.position[0] += 0.5 * particle.v[0] * dt;

            // Step 2: Find iCell and remainder for interpolation
            double x_norm = particle.position[0] / this->layout_->cell_size(Direction::X);
            int iCell = static_cast<int>(x_norm) + this->layout_->dual_dom_start(Direction::X);
            double remainder = x_norm - floor(x_norm);

            // Step 3: Interpolate E and B fields at the particle position
            double Ex = interpolate(E.x, iCell, remainder);
            double Ey = interpolate(E.y, iCell, remainder);
            double Ez = interpolate(E.z, iCell, remainder);

            double Bx = interpolate(B.x, iCell, remainder);
            double By = interpolate(B.y, iCell, remainder);
            double Bz = interpolate(B.z, iCell, remainder);

            // Step 4: Half acceleration due to E field, defining v_minus
            double qmdt2 = 0.5 * dt * (particle.charge / particle.mass); 

            double vx_minus = particle.v[0] + qmdt2 * Ex;
            double vy_minus = particle.v[1] + qmdt2 * Ey;
            double vz_minus = particle.v[2] + qmdt2 * Ez;

            // Step 5: Rotation due to B field, defining t, s, v' = v_minus + v_minus x t , v_plus = v_minus + v' x s
            double tx = qmdt2 * Bx;
            double ty = qmdt2 * By;
            double tz = qmdt2 * Bz;

            double t2 = tx * tx + ty * ty + tz * tz;

            double sx = 2.0 * tx / (1.0 + t2);
            double sy = 2.0 * ty / (1.0 + t2);
            double sz = 2.0 * tz / (1.0 + t2);

            double vpx = vx_minus + (vy_minus * tz - vz_minus * ty);
            double vpy = vy_minus + (vz_minus * tx - vx_minus * tz);
            double vpz = vz_minus + (vx_minus * ty - vy_minus * tx);

            double vx_plus = vx_minus + (vpy * sz - vpz * sy);
            double vy_plus = vy_minus + (vpz * sx - vpx * sz);
            double vz_plus = vz_minus + (vpx * sy - vpy * sx);

            // Step 6: Second half acceleration due to E
            particle.v[0] = vx_plus + qmdt2 * Ex;
            particle.v[1] = vy_plus + qmdt2 * Ey;
            particle.v[2] = vz_plus + qmdt2 * Ez;

            // Step 7: Second half-step position update
            particle.position[0] += 0.5 * particle.v[0] * dt; 
            
            
        }
    }

private:
    double interpolate(Field<dimension> const& field, int iCell, double remainder) const
    {
        if (this->layout_->centerings(field.quantity())[0] == this->layout_->dual)
        {
            if (remainder < 0.5)
                return field(iCell - 1) * (1.0 - remainder) + field(iCell) * remainder;
            else
                return field(iCell) * (1.0 - remainder) + field(iCell + 1) * remainder;
        }
        return field(iCell) * (1.0 - remainder) + field(iCell + 1) * remainder;
    }
};


#endif
