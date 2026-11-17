##############################################
# Programmer: Adrien Dumont
# Class: ENSC 201-01, Fall 2026
# Programming Assignment #7
# 11/17/26
#
# Description: Damped simple pendulum simulation. Reads simulation
# parameters from pendulum_parameters.txt, steps the pendulum forward in
# time with semi-implicit Euler integration, counts equilibrium crossings
# and complete swings, stops when the angular velocity at an equilibrium
# crossing drops to the threshold (or when total_time is reached), and
# writes pendulum_trajectory.csv and pendulum_summary.txt.
#
# Notes: Written with help from Claude (Anthropic AI assistant). Physics
# model and threshold/swing rules come from the PA7 specification.
##############################################
import math
import os

# build paths from this script's folder so the program works no matter
# which directory it is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMETERS_FILENAME = os.path.join(SCRIPT_DIR, "pendulum_parameters.txt")
TRAJECTORY_FILENAME = os.path.join(SCRIPT_DIR, "pendulum_trajectory.csv")
SUMMARY_FILENAME = os.path.join(SCRIPT_DIR, "pendulum_summary.txt")

# stopping reasons reported in the summary
REASON_THRESHOLD = "angular velocity at an equilibrium crossing reached " \
    "the threshold"
REASON_TIME = "total_time was reached before the threshold"


def read_parameters(filename):
    """Reads simulation parameters from a text file into a dictionary.

    Args:
        filename (str): name of the parameter file to read

    Returns:
        dict: parameter name (str) -> parameter value (float)

    Notes:
        Each line holds one name-value pair separated by "=", ":" or ",".
        Blank lines, lines starting with "#", and lines whose value is not
        a number (such as a header row) are skipped.
    """
    parameters = {}

    infile = open(filename, "r", encoding="utf-8-sig")
    for line in infile:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue

        # split on the first separator found in the line
        for separator in ["=", ":", ","]:
            if separator in line:
                name, value = line.split(separator, 1)
                try:
                    parameters[name.strip()] = float(value.strip())
                except ValueError:
                    pass  # not a numeric line (e.g., a header), skip it
                break
    infile.close()

    return parameters


def compute_angular_acceleration(angle, gravity, length):
    """Computes the pendulum's angular acceleration at a given angle.

    Args:
        angle (float): current angle from vertical in radians
        gravity (float): gravitational acceleration in m/s^2
        length (float): pendulum length in meters

    Returns:
        float: angular acceleration in rad/s^2

    Notes:
        Pre-conditions:
            - length > 0
    """
    angular_acceleration = -(gravity / length) * math.sin(angle)
    return angular_acceleration


def step_pendulum(angle, angular_velocity, gravity, length,
                  friction_coefficient, time_step):
    """Advances the pendulum state by one time step (semi-implicit Euler).

    Args:
        angle (float): current angle in radians
        angular_velocity (float): current angular velocity in rad/s
        gravity (float): gravitational acceleration in m/s^2
        length (float): pendulum length in meters
        friction_coefficient (float): unitless decay factor applied to the
            angular velocity each step (closer to 1 = less damping)
        time_step (float): time step in seconds

    Returns:
        float: the next angle in radians
        float: the next angular velocity in rad/s

    Notes:
        The next angle is calculated with the NEXT angular velocity
        (semi-implicit Euler), which keeps the simulation stable.
    """
    angular_acceleration = compute_angular_acceleration(angle, gravity,
                                                        length)
    next_angular_velocity = (angular_velocity * friction_coefficient
                             + angular_acceleration * time_step)
    next_angle = angle + next_angular_velocity * time_step
    return next_angle, next_angular_velocity


def is_equilibrium_crossing(previous_angle, current_angle):
    """Checks whether the pendulum crossed equilibrium (angle changed sign).

    Args:
        previous_angle (float): angle at the previous step in radians
        current_angle (float): angle at the current step in radians

    Returns:
        bool: True if the angle changed sign, False otherwise

    Notes:
        A product of exactly zero (an angle sitting at 0) is not a crossing.
    """
    return previous_angle * current_angle < 0


def is_threshold_met(angular_velocity, threshold):
    """Checks whether the angular velocity magnitude is at/below a threshold.

    Args:
        angular_velocity (float): angular velocity in rad/s
        threshold (float): minimum angular velocity threshold in rad/s

    Returns:
        bool: True if abs(angular_velocity) <= threshold, False otherwise
    """
    return abs(angular_velocity) <= threshold


def count_swings(crossings):
    """Counts complete swings from the number of equilibrium crossings.

    Args:
        crossings (int): number of equilibrium crossings

    Returns:
        int: number of complete swings (two crossings per swing)

    Notes:
        Pre-conditions:
            - crossings >= 0
    """
    return crossings // 2


def simulate_pendulum(parameters):
    """Runs the damped pendulum time-stepping simulation.

    Args:
        parameters (dict): simulation parameters with the keys
            initial_angle_degrees, length, gravity, friction_coefficient,
            time_step, total_time, angular_velocity_threshold

    Returns:
        dict: simulation results with the keys
            times (list of float): time of each recorded state in seconds
            angles (list of float): angle of each state in radians
            angular_velocities (list of float): angular velocity of each
                state in rad/s
            crossings (int): number of equilibrium crossings
            swings (int): number of complete swings
            threshold_time (float or None): time the threshold was reached,
                or None if it was not reached
            final_time (float): time of the last recorded state
            stop_reason (str): why the simulation stopped

    Notes:
        The pendulum starts from rest at the initial angle. The threshold
        is only checked at equilibrium crossings, because the angular
        velocity is naturally near zero at every turning point.
    """
    gravity = parameters["gravity"]
    length = parameters["length"]
    friction_coefficient = parameters["friction_coefficient"]
    time_step = parameters["time_step"]
    total_time = parameters["total_time"]
    threshold = parameters["angular_velocity_threshold"]

    # initial state: released from rest, angle converted to radians
    angle = math.radians(parameters["initial_angle_degrees"])
    angular_velocity = 0.0
    times = [0.0]
    angles = [angle]
    angular_velocities = [angular_velocity]

    crossings = 0
    threshold_time = None
    stop_reason = REASON_TIME

    # compute the step count up front so repeated float additions of
    # time_step cannot add or drop a step at the end
    num_steps = round(total_time / time_step)

    for step in range(1, num_steps + 1):
        previous_angle = angle
        angle, angular_velocity = step_pendulum(angle, angular_velocity,
                                                gravity, length,
                                                friction_coefficient,
                                                time_step)
        time = step * time_step

        # record this step
        times.append(time)
        angles.append(angle)
        angular_velocities.append(angular_velocity)

        # only check the threshold when the pendulum crosses equilibrium
        if is_equilibrium_crossing(previous_angle, angle):
            crossings += 1
            if is_threshold_met(angular_velocity, threshold):
                threshold_time = time
                stop_reason = REASON_THRESHOLD
                break

    results = {
        "times": times,
        "angles": angles,
        "angular_velocities": angular_velocities,
        "crossings": crossings,
        "swings": count_swings(crossings),
        "threshold_time": threshold_time,
        "final_time": times[-1],
        "stop_reason": stop_reason,
    }
    return results


def write_trajectory(filename, times, angles, angular_velocities):
    """Writes the time-series trajectory data to a CSV file.

    Args:
        filename (str): name of the CSV file to write
        times (list of float): time of each state in seconds
        angles (list of float): angle of each state in radians
        angular_velocities (list of float): angular velocity of each state
            in rad/s

    Notes:
        Post-conditions:
            - the file is overwritten with a header row and one row per
              recorded state
    """
    outfile = open(filename, "w")
    outfile.write("time_s,angle_rad,angular_velocity_rad_s\n")
    for i in range(len(times)):
        outfile.write(f"{times[i]:.2f},{angles[i]:.6f},"
                      f"{angular_velocities[i]:.6f}\n")
    outfile.close()


def write_summary(filename, parameters, results):
    """Writes a human-readable simulation summary report to a text file.

    Args:
        filename (str): name of the text file to write
        parameters (dict): the simulation parameters that were used
        results (dict): the results returned by simulate_pendulum()

    Notes:
        Post-conditions:
            - the file is overwritten with the parameter values, time to
              threshold, complete swings, final time, and stopping reason
    """
    if results["threshold_time"] is None:
        threshold_text = "not reached"
    else:
        threshold_text = f"{results['threshold_time']:.2f} s"

    outfile = open(filename, "w")
    outfile.write("Damped Pendulum Simulation Summary\n")
    outfile.write("==================================\n")
    outfile.write("Input parameters:\n")
    for name in parameters:
        outfile.write(f"  {name}: {parameters[name]}\n")
    outfile.write("\nResults:\n")
    outfile.write(f"  Time to threshold: {threshold_text}\n")
    outfile.write(f"  Equilibrium crossings: {results['crossings']}\n")
    outfile.write(f"  Complete swings: {results['swings']}\n")
    outfile.write(f"  Final simulation time: {results['final_time']:.2f} s\n")
    outfile.write(f"  Stopping reason: {results['stop_reason']}\n")
    outfile.close()


def main():
    """Coordinates the workflow: read parameters, simulate, write outputs.
    """
    parameters = read_parameters(PARAMETERS_FILENAME)
    results = simulate_pendulum(parameters)

    write_trajectory(TRAJECTORY_FILENAME, results["times"],
                     results["angles"], results["angular_velocities"])
    write_summary(SUMMARY_FILENAME, parameters, results)

    print(f"Simulation stopped at {results['final_time']:.2f} s "
          f"({results['stop_reason']}).")
    print(f"Wrote {TRAJECTORY_FILENAME}")
    print(f"Wrote {SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()
