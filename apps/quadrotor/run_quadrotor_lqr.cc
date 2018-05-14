/// @file
///
/// This demo sets up a controlled Quadrotor that uses a Linear Quadratic
/// Regulator to (locally) stabilize a nominal hover.

#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "quadrotor_plant.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "trajectory_controller.h"

DEFINE_int32(simulation_trials, 10, "Number of trials to simulate.");
DEFINE_double(simulation_real_time_rate, 1.0, "Real time rate");
DEFINE_double(trial_duration, 7.0, "Duration of execution of each trial");

namespace drake {
using systems::DiagramBuilder;
using systems::Simulator;
using systems::Context;
using systems::ContinuousState;
using systems::VectorBase;
using systems::TrajectorySource;
using namespace trajectories;

namespace examples {
namespace quadrotor {
namespace {

PiecewisePolynomial<double> MinimumSnap(const Eigen::VectorXd &breaks, const Eigen::MatrixXd &knots, int order, int k_r, int k_phi)
{
    double mu_r = 1.0;
    double mu_phi = 1.0;
    MatrixXd poly_r = MatrixXd::Zero(order + 1, 1);
    MatrixXd poly_phi = MatrixXd::Zero(order + 1, 1);
    for (int i = k_r; i <= order; i++)
    {
        int term = 1;
        for (int j = 0; j < k_r; j++)
        {
            term = term * (i - j);
        }
        poly_r(order - i, 0) = (double)term;
    }
    for (int i = k_phi; i <= order; i++)
    {
        int term = 1;
        for (int j = 0; j < k_phi; j++)
        {
            term = term * (i - j);
        }
        poly_phi(order - i, 0) = (double)term;
    }
    MatrixXd Q(4 * (order + 1) * (breaks.size() - 1), 4 * (order + 1) * (breaks.size() - 1));
    for (int i = 0; i < breaks.size() - 1; i++)
    {
        MatrixXd Q_r = MatrixXd::Zero(order + 1, order + 1);
        for (int j = 0; j <= order - k_r; j++)
        {
            for (int k = j; k <= order - k_r; k++)
            {
                int order_t_r = order - k_r - j + order - k_r - k;
                if (j == k)
                {
                    Q_r(j, k) = poly_r(j) * poly_r(j) / (order_t_r + 1) * (std::pow(breaks(i + 1), order_t_r + 1) - std::pow(breaks(i), order_t_r + 1));
                }
                else
                {
                    Q_r(j, k) = 2 * poly_r(j) * poly_r(k) / (order_t_r + 1) * (std::pow(breaks(i + 1), order_t_r + 1) - std::pow(breaks(i), order_t_r + 1));
                }
            }
        }
        MatrixXd Q_phi = MatrixXd::Zero(order + 1, order + 1);
        for (int j = 0; j <= order - k_phi; j++)
        {
            for (int k = j; k <= order - k_phi; k++)
            {
                int order_t_r = order - k_phi - j + order - k_phi - k;
                if (j == k)
                {
                    Q_phi(j, k) = poly_phi(j) * poly_phi(j) / (order_t_r + 1) * (std::pow(breaks(i + 1), order_t_r + 1) - std::pow(breaks(i), order_t_r + 1));
                }
                else
                {
                    Q_phi(j, k) = 2 * poly_phi(j) * poly_phi(k) / (order_t_r + 1) * (std::pow(breaks(i + 1), order_t_r + 1) - std::pow(breaks(i), order_t_r + 1));
                }
            }
        }
        Q.block(i * 4 * (order + 1), i * 4 * (order + 1), order + 1, order + 1) = mu_r * Q_r;
        Q.block(i * 4 * (order + 1) + order + 1, i * 4 * (order + 1) + order + 1, order + 1, order + 1) = mu_r * Q_r;
        Q.block(i * 4 * (order + 1) + 2 * (order + 1), i * 4 * (order + 1) + 2 * (order + 1), order + 1, order + 1) = mu_r * Q_r;
        Q.block(i * 4 * (order + 1) + 3 * (order + 1), i * 3 * (order + 1) + 2 * (order + 1), order + 1, order + 1) = mu_phi * Q_phi;
    }
    std::cout << Q << std::endl;
    MatrixXd C1 = MatrixXd::Zero((knots.cols() * 2 - 2) * 4, 4 * (order + 1) * (breaks.size() - 1));
    VectorXd b1 = VectorXd::Zero((knots.cols() * 2 - 2) * 4);
    for (int i = 0; i < knots.cols() - 1; i++)
    {
        RowVectorXd time_values(order + 1);
        for (int j = 0; j <= order; j++)
        {
            time_values(order - j) = std::pow(breaks(i), j);
        }
        for (int j = 0; j < 4; j++)
        {
            RowVectorXd c = RowVectorXd::Zero(4 * (order + 1) * (breaks.size() - 1));
            c.segment(i * (order + 1) * 4  + j * (order + 1), order + 1) = time_values;
            C1.row(8 * i + j) = c;
        }
        b1.segment(8 * i, 4) = knots.col(i);
        for (int j = 0; j <= order; j++)
        {
            time_values(order - j) = std::pow(breaks(i + 1), j);
        }
        for (int j = 0; j < 4; j++)
        {
            RowVectorXd c = RowVectorXd::Zero(4 * (order + 1) * (breaks.size() - 1));
            c.segment(i * (order + 1) * 4  + j * (order + 1), order + 1) = time_values;
            C1.row(8 * i + 4 + j) = c;
        }
        b1.segment(8 * i + 4, 4) = knots.col(i + 1);
    }
    MatrixXd C2 = MatrixXd::Zero(Dynamic, 4 * (order + 1) * (breaks.size() - 1));
    VectorXd b2 = VectorXd::Zero(Dynamic);
    MatrixXf continuous = MatrixXf::Zero(breaks.size(), k_r);
    continuous.block(1, 0, breaks.size() - 2, k_r) = 1;
    for (int i = 0; i < knots.cols(); i++)
    {
        for (int j = 0; j < k_r; j++)
        {
            RowVectorXd time_values(order + 1) = RowVectorXd::Zero(order + 1);
            RowVectorXd polyder = RowVectorXd::Zero(order + 1);
            for (int k = k_r; k <= order; k++)
            {
                int term = 1;
                for (int l = 0; l <= j; l++)
                {
                    term = term * (k - l);
                }
                time_values(order - k) = std::pow(breaks(i), k - k_r) * (double)term;
            }
            for (int k = 0; k < 3; k++)
            {
                if (continuous(i, j) == 1)
                {
                    RowVectorXd c = RowVectorXd::Zero(4 * (order + 1) * (breaks.size() - 1));
                    c.segment(i * (order + 1) * 4  + k * (order + 1), order + 1) = time_values;
                    c.segment((i - 1) * (order + 1) * 4  + k * (order + 1), order + 1) = -1 * time_values;
                    C2 << c;
                    b2 << 0;
                }
                else
                {
                    if (i != breaks.size() - 1)
                    {
                        RowVectorXd c = RowVectorXd::Zero(4 * (order + 1) * (breaks.size() - 1));
                        c.segment(i * (order + 1) * 4  + k * (order + 1), order + 1) = time_values;
                        C2 << c;
                        b2 << 0;
                    }
                    if (i != 0)
                    {
                        RowVectorXd c = RowVectorXd::Zero(4 * (order + 1) * (breaks.size() - 1));
                        c.segment((i - 1) * (order + 1) * 4  + k * (order + 1), order + 1) = time_values;
                        C2 << c;
                        b2 << 0;
                    }
                }
            }
        }
    }
    MatrixXd C(Dynamic, 4 * (order + 1) * (breaks.size() - 1));
    VectorXd b(Dynamic);
    C << C1, C2;
    b << b1, b2;
    MathematicalProgram prog;
    auto x = prog.NewContinuousVariables(4 * (order + 1) * (breaks.size() - 1), "x");
    prog.AddQuadraticCost(Q, VectorXd::Zero(4 * (order + 1) * (breaks.size() - 1)), x);
    prog.AddLinearEqualityConstraint(C, b, x);
    prog.Solve();
    VectorXd x_sol = prog.GetSolution(x);
    std::vector<PolynomialMatrix> polynomials(breaks.size() - 1);
    for (int i = 0; i < breaks.size() - 1; i++)
    {
        polynomials[i].resize(4, 1);
    }
    for (int i = 0; i < breaks.size() - 1; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            Polynomial<double> poly = Polynomial<double>(x_sol.segment(i * (order + 1) * 4 + j * (order + 1), order + 1).reverse());
            polynomials[i](j, 0) = poly:
        }
    }
    return PiecewisePolynomial<double>(polynomials, breaks);
}

int do_main() {
  lcm::DrakeLcm lcm;

  DiagramBuilder<double> builder;

  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/quadrotor.urdf"),
      multibody::joints::kRollPitchYaw, tree.get());

  // The nominal hover position is at (0, 0, 1.0) in world coordinates.
  const Eigen::Vector3d kNominalPosition{((Eigen::Vector3d() << 0.0, 0.0, 1.0).
      finished())};

  auto quadrotor = builder.AddSystem<QuadrotorPlant<double>>();
  quadrotor->set_name("quadrotor");
  auto controller = builder.AddSystem(std::make_unique<TrajectoryController<double>>(*quadrotor, 1.5, 3.0, 1.0, 0.8));
  controller->set_name("controller");
  auto visualizer =
      builder.AddSystem<drake::systems::DrakeVisualizer>(*tree, &lcm);
  visualizer->set_name("visualizer");
  int num_knots = 5;
  VectorXd breaks(num_knots);
  breaks << 0.0, 5.0, 10.0, 15.0, 20.0;
  MatrixXd knots(4, num_knots);
  Vector4d knot1(3, 3, 1, 0);
  Vector4d knot2(-3, 3, 1, M_PI / 2.);
  Vector4d knot3(-3, -3, 1, M_PI);
  Vector4d knot4(3, -3, 1, 3 * M_PI / 2.);
  Vector4d knot5(3, 3, 1, 0);
  knots << knot1, knot2, knot3, knot4, knot5;
  const PiecewisePolynomial<double> traj = MinimumSnap(breaks, knots, 6, 4, 4);
  auto trajectory = builder.AddSystem<systems::TrajectorySource>(traj);
  trajectory->set_name("trajectory");

  builder.Connect(trajectory->get_output_port(), controller->get_input_port(1));
  builder.Connect(quadrotor->get_output_port(0), controller->get_input_port(0));
  builder.Connect(controller->get_output_port(0), quadrotor->get_input_port(0));
  builder.Connect(quadrotor->get_output_port(0), visualizer->get_input_port(0));

  auto diagram = builder.Build();
  Simulator<double> simulator(*diagram);
  VectorX<double> x0 = VectorX<double>::Zero(12);

  //x0 = VectorX<double>::Random(12);

  simulator.get_mutable_context()
      .get_mutable_continuous_state_vector()
      .SetFromVector(x0);

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_simulation_real_time_rate);
  simulator.StepTo(traj.end_time() + 5.0);

  return 0;
}

}  // namespace
}  // namespace quadrotor
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::quadrotor::do_main();
}
