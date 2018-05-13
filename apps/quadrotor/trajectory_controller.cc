#include "trajectory_controller.h"

#include <string>

#include "drake/common/default_scalars.h"
#include "drake/math/rotation_matrix.h"
#include "drake/util/drakeGeometryUtil.h"

namespace drake {
using namespace systems;
namespace examples {
namespace quadrotor {

template <typename T>
TrajectoryController<T>::TrajectoryController(const QuadrotorPlant<T>& system,
                const T kP, const T kV, const T kR, const T kW)
    : kP(kP),
    kV(kV),
    kR(kR),
    kW(kW),
    m_(system.m()),
    g_(system.g())
{
    this->DeclareVectorOutputPort(BasicVector<T>(system.get_input_size()), &TrajectoryController<T>::ComputeInput);
    this->DeclareInputPort(kVectorValued, system.get_num_states());
    this->DeclareInputPort(kVectorValued, 4);
    double kF = system.kF();
    double kM = system.kM();
    double L = system.L();
    invert_dynamics << kF, kF, kF, kF,
                    0, kF * L, 0, -kF * L,
                    -kF * L, 0, kF * L, 0,
                    kM, -kM, kM, -kM;
    invert_dynamics = invert_dynamics.inverse();
}

template <typename T>
TrajectoryController<T>::~TrajectoryController() {}

template <typename T>
void TrajectoryController<T>::ComputeInput(const Context<T>& context, BasicVector<T>* input) const
{
    VectorX<T> state = this->EvalVectorInput(context, 0)->get_value();
    VectorX<T> state_d = this->EvalVectorInput(context, 1)->get_value();
    Vector3<T> F_des = -kP * (state.head(3) - state_d.head(3)) - kV * (state.segment(6, 3)) + m_ * g_ * Vector3<T>(0, 0, 1.0);
    T phi = state(5);
    T phi_d = state_d(3);
    Vector3<T> rpy = state.segment(3, 3);
    Vector3<T> rpy_dot = state.segment(9, 3);
    Matrix3<T> rpy_mat = drake::math::rpy2rotmat(rpy);
    Vector3<T> z_B = rpy_mat * Vector3<T>(0, 0, 1.0);
    Vector3<T> x_C = Vector3<T>(std::cos(phi), std::sin(phi), 0);
    Vector3<T> y_B = z_B.cross(x_C).normalized();
    Vector3<T> x_B = y_B.cross(z_B);
    Vector3<T> z_B_des = F_des.normalized();
    Vector3<T> x_C_des = Vector3<T>(std::cos(phi_d), std::sin(phi_d), 0);
    Vector3<T> y_B_des = z_B_des.cross(x_C_des).normalized();
    Vector3<T> x_B_des = y_B_des.cross(z_B_des);
    Matrix3<T> R_B;
    R_B << x_B, y_B, z_B;
    Matrix3<T> R_des;
    R_des << x_B_des, y_B_des, z_B_des;
    Vector3<T> e_R = 0.5 * this->vee(R_des.transpose() * R_B - R_B.transpose() * R_des);
    Vector3<T> omega;
    rpydot2angularvel(rpy, rpy_dot, omega);
    omega = rpy_mat.adjoint() * omega;
    Vector4<T> u;
    u << F_des.dot(z_B), -kR * e_R - kW * omega;
    input->SetFromVector(invert_dynamics * u);
}

template <typename T>
Vector3<T> TrajectoryController<T>::vee(const Matrix3<T>& mat) const
{
    return 0.5 * Vector3<T>(mat(2, 1) - mat(1, 2),
                    mat(0, 2) - mat(2, 0),
                    mat(1, 0) - mat(0, 1));
}

}
}
}

template class ::drake::examples::quadrotor::TrajectoryController<double>;
