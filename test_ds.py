import slideflow as sf, openslide
import pyvips
from pathlib import Path

path = Path("/exports/path-pulmogroep-hpc/Jurre/SalvDataset/LUMC_cohort/slides/LUMC-T11_00874-1B-HE-000.mrxs")
#path = "/exports/path-pulmogroep-hpc/Jurre/SalvDataset/LUMC_cohort/slides/LUMC-T11_00875-1_1_1-HE-000.tiff"

import openslide
slide = openslide.OpenSlide(path)
print("openslide")
print("MPP X:", slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
print("MPP Y:", slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))


img = pyvips.Image.new_from_file(path, access="sequential")
print("All libvips metadata fields:")
print(img.get_fields())

img = pyvips.Image.new_from_file(path, access="sequential")
print("pyvips")
print("openslide-mpp-x:", img.get("openslide.mpp-x"))
print("openslide-mpp-y:", img.get("openslide.mpp-y"))