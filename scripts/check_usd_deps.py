from pxr import Usd, UsdUtils
refs = UsdUtils.ComputeAllDependencies('c:/Users/agni_/Documents/hideNseek/data/scenes/hospital/hospital.usd')
for r in refs[0][:50]:
    print(r)
