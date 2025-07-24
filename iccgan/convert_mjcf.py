from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import omni.kit.commands
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools

# create new stage
omni.usd.get_context().new_stage()

# setting up import configuration:
status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
import_config.set_fix_base(False)
import_config.set_make_default_prim(False)

# Get path to extension data:
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
extension_path = ext_manager.get_extension_path(ext_id)

# import MJCF
omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path=extension_path + "assets/humanoid.xml",
    import_config=import_config,
    prim_path="/World/humanoid"
)

# get stage handle
stage = omni.usd.get_context().get_stage()

# enable physics
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))

# set gravity
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(981.0)

# add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)

print("DONE3")