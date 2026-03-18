"""Print USD scene dependencies. Usage: python check_usd_deps.py <scene.usd>"""
import sys
from pxr import Usd, UsdUtils

scene = sys.argv[1] if len(sys.argv) > 1 else "data/scenes/hospital/hospital.usd"
refs = UsdUtils.ComputeAllDependencies(scene)
for r in refs[0][:50]:
    print(r)
