from isaaclab.app import AppLauncher
import os

# 1) Launch headless Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 2) Imports for commands & USD APIs
import omni.kit.commands
import omni.usd                                        # <-- make sure you have omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import UsdPhysics
import isaacsim.core.utils.articulations as articulations_utils
import isaacsim.core.utils.prims as prims_utils

# 3) Turn on the MJCF importer
enable_extension("isaacsim.asset.importer.mjcf")

# 4) New blank stage
omni.usd.get_context().new_stage()

# 5) Build import config
status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
# Floating‑base: keep the freejoint instead of welding pelvis to the world
import_config.set_fix_base(False)
import_config.set_make_default_prim(False)

# 6) Run the import
mjcf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.xml")
omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path=mjcf_path,
    import_config=import_config,
    prim_path="/World/humanoid"        # your new root Xform
)

# 7) Mark the pelvis as articulation root
stage = omni.usd.get_context().get_stage()
humanoid_prim = prims_utils.get_prim_at_path(prim_path="/World/humanoid")
articulations_utils.add_articulation_root(prim=humanoid_prim)

# 8) Save it out
output_usd = os.path.join(os.path.dirname(__file__), "assets", "humanoid.usd")
omni.usd.get_context().save_as_stage(output_usd)
print(f"Saved USD to {output_usd}")

# 9) Clean up
simulation_app.close()
