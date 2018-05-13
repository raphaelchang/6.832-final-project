#pragma once

#include <memory>

#include <Eigen/Core>

#include "drake/systems/framework/leaf_system.h"
#include "quadrotor_plant.h"
#include "drake/common/drake_copyable.h"

using namespace Eigen;
namespace drake {
namespace examples {
namespace quadrotor {

template <typename T>
class TrajectoryController final : public systems::LeafSystem<T> {
    public:
        DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TrajectoryController)
        TrajectoryController(const QuadrotorPlant<T>& system,
                const T kP, const T kV, const T kR, const T kW);

        ~TrajectoryController() override;
    private:
        void ComputeInput(const systems::Context<T>& context, systems::BasicVector<T>* input) const;
        Vector3<T> vee(const Matrix3<T>& mat) const;

        const T kP;
        const T kV;
        const T kR;
        const T kW;
        const T m_;
        const T g_;

        Matrix<T, 4, 4> invert_dynamics;
};

}
}
}
