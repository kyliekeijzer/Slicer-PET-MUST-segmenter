cmake_minimum_required(VERSION 3.13.4)

project(MUSTsegmenter)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MUSTsegmenter")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Kylie Keijzer (University Medical Center Groningen (UMCG))")
set(EXTENSION_DESCRIPTION "Multiple SUV threshold segmenter tool")
set(EXTENSION_ICONURL "https://github.com/kyliekeijzer/Slicer-MUST-segmenter/blob/master/Slicer-MUST-segmenter.png?raw=true")
set(EXTENSION_SCREENSHOTURLS "https://github.com/kyliekeijzer/Slicer-MUST-segmenter/blob/master/screenshots/Slicer-MUST-segmenter_screenshot.png?raw=true")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MUSTSegmenter)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
