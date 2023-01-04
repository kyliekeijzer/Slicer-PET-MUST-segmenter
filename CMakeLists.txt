cmake_minimum_required(VERSION 3.13.4)

project(PET-MUST-Segmenter)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/kyliekeijzer/Slicer-PET-MUST-segmenter.git")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Kylie Keijzer (University Medical Center Groningen (UMCG))")
set(EXTENSION_DESCRIPTION "Multiple SUV Thresholding (MUST)-segmenter is a semi-automated PET image segmentation tool that enables delineation of multiple lesions at once, and extracts the lesions' features")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/kyliekeijzer/Slicer-PET-MUST-segmenter/master/Slicer-MUST-segmenter.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/kyliekeijzer/Slicer-PET-MUST-segmenter/master/screenshots/Slicer-MUST-segmenter_screenshot.png https://raw.githubusercontent.com/kyliekeijzer/Slicer-PET-MUST-segmenter/master/screenshots/9.png https://raw.githubusercontent.com/kyliekeijzer/Slicer-PET-MUST-segmenter/master/screenshots/12.png https://raw.githubusercontent.com/kyliekeijzer/Slicer-PET-MUST-segmenter/master/screenshots/10.png https://raw.githubusercontent.com/kyliekeijzer/Slicer-PET-MUST-segmenter/master/screenshots/25.png")
set(EXTENSION_DEPENDS "SlicerRadiomics") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MUSTsegmenter)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
