from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext

def main():

    """Main function."""
    sim_cfg = SimulationCfg(dt=0.01)

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.reset()

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()