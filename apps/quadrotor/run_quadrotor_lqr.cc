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
  auto controller = builder.AddSystem(std::make_unique<TrajectoryController<double>>(*quadrotor, 1.0, 2.0, 1.0, 0.8));
  controller->set_name("controller");
  auto visualizer =
      builder.AddSystem<drake::systems::DrakeVisualizer>(*tree, &lcm);
  visualizer->set_name("visualizer");
  Vector2d breaks;
  breaks << 0.0, 10.0;
  MatrixXd knots(4, 2);
  knots << 1, 1, 3, 3, 1.0, 1.0, 0.5, 0.5;
  const PiecewisePolynomial<double> traj = PiecewisePolynomial<double>::ZeroOrderHold(breaks, knots);
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
  simulator.StepTo(traj.end_time());

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
