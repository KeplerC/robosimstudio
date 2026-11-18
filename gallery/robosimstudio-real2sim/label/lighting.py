"""Place the path tracer's key at the modeled workshop window.

The key and material UVs are fixed at renderer construction. No geometry
or simulation state is changed; room surfaces remain scene geometry.
"""

from robosimstudio.core.render.scenery import register_scenery


@register_scenery("label_window_light")
def window_light(kit):
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    stage = kit.stage
    xf = UsdGeom.Xform.Get(stage, "/root/_LabPanel")
    xf.ClearXformOpOrder()
    source = Gf.Vec3d(kit.cell.center[0] - 0.87, kit.cell.center[1] - 1.12, kit.surface_z + 1.122)
    aim = Gf.Vec3d(kit.cell.center[0], kit.cell.center[1] - 0.15, kit.surface_z + 0.022)
    direction = (aim - source).GetNormalized()
    xf.AddTranslateOp().Set(source)
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, -1), direction).GetQuat()))
    light = UsdLux.RectLight.Get(stage, "/root/_LabPanel/light")
    light.GetWidthAttr().Set(0.68)
    light.GetHeightAttr().Set(0.60)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.90, 0.80))
    light.GetIntensityAttr().Set(light.GetIntensityAttr().Get() * 3.0)

    # The general cloth renderer projects the first rendered shape by PCA.
    # Its principal-axis signs depend on that pose. A shipping print instead
    # has fixed material coordinates in the tape's unchanged 9 x 20 grid.
    # Set these once on the real deforming mesh, never as a re-posed decal.
    uv = [Gf.Vec2f(0.095 - j * 0.005, 0.040 - i * 0.005) for j in range(20) for i in range(9)]
    for prim in stage.Traverse():
        if str(prim.GetPath()).endswith("/model/deformables/label"):
            pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            pv.Set(uv)
