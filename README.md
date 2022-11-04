# MUST-segmenter
***MU***ltiple ***S***UV ***T***hresholding (MUST)-segmenter is a semi-automated PET image segmentation tool that enables delineation of 
multiple lesions at once, and extracts metabolic active tumor volume (MATV).

![](screenshots/Slicer-MUST-segmenter_screenshot.png?raw=true "MUST-segmenter screenshot")  

## 1. Installation
Download Slicer V4.11 [here](https://slicer-packages.kitware.com/#collection/5f4474d0e1d8c75dfc70547e/folder/60ac0ce2ae4540bf6a899ecc)  
tbd.

## 2. Tutorial
#### DEMO video tbd.
<br>

### Preparation

1. [Import PET and CT DICOM series into 3D Slicer ](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html#dicom-import)  
This can be retrieved by drag and dropping files into the DICOM database or by importing files into the DICOM database.
   - (optional) Import avoidance regions and brain contours as ***[Segmentations](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmentations.html#import-an-existing-segmentation-from-volume-file)***.

2. [Load patient data into the scene](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html#dicom-loading)  
Load ***one*** patient's PET and CT images (with or without avoidance regions and brain contours).  
![](screenshots/1.png?raw=true "")  
3. Display PET as overlay on CT
![](screenshots/3.png?raw=true "")  
4. Edit Window/Level display of PET and CT images and set to preferred settings  
a. ![](screenshots/4.png?raw=true "")  
b. ![](screenshots/5.png?raw=true "")  
5. Go to MUST-segmenter  
![](screenshots/6.png?raw=true "")  

### Segmentation with MUST-segmenter
1. Identify tumors by seed placing (Markups)  
   ![](screenshots/8.png?raw=true "") &rarr; ![](screenshots/9.png?raw=true "")  
2. Choose segmentation method(s):  
   ![](screenshots/10.png?raw=true "")  
   - Fixed thresholds: **SUV 2.5**, **SUV 3.0** and **SUV 4.0**  
   - **41% SUVmax**  
     a. Place regions of interest (ROIs) around the identified tumors (the seeds don't need to be removed)  
        ![](screenshots/11.png?raw=true "") &rarr; ![](screenshots/12.png?raw=true "")  
     b. (optional) Take a threshold for each individual ROI  
        ![](screenshots/13.png?raw=true "")  
   - Liver-based thresholds: **Liver SUVmax** and **PERCIST**  
     a. Create a new MarkupsFiducial and rename it to 'liver'  
        ![](screenshots/14.png?raw=true "") &rarr; ![](screenshots/15.png?raw=true "") &rarr; ![](screenshots/16.png?raw=true "") &rarr; ![](screenshots/17.png?raw=true "")  
     b. Place the seed in the right lobe of the liver  
        ![](screenshots/18.png?raw=true "")  
     c. Create the liver sphere  
        ![](screenshots/19.png?raw=true "") &rarr; ![](screenshots/20.png?raw=true "")  
     d. *Delete* the 'liver' MarkupsFiducial!  
        ![](screenshots/21.png?raw=true "")  
   - Majority Voting: **MV2** and **MV3**  
     a. Make sure the following segmentation methods are *additionally* selected:  
        *SUV 2.5, SUV 4.0, 41% SUVmax, Liver SUVmax and PERCIST*  
   - **Brain region based**  
     a. Provide a brain segmentation into the scene as a segmentation  
        ![](screenshots/22.png?raw=true "")  
     b. Rename the brain segmentation node to 'brain'  
        ![](screenshots/23.png?raw=true "") &rarr; ![](screenshots/7.png?raw=true "")  
3. Perform segmentation and wait for the results!  
   ![](screenshots/24.png?raw=true "") &rarr; ![](screenshots/25.png?raw=true "")  
4. Save the segmentations in the desired format  
   ![](screenshots/33.png?raw=true "") &rarr; ![](screenshots/34.png?raw=true "")  
   Before exporting to NRRD or NIFTI, the segmentation node needs to be converted to a labelmap node!  
   ![](screenshots/35.png?raw=true "")  

### MATV extraction
1. Select the segmentation results that are present in the scene  
![](screenshots/10.png?raw=true "")  
2. Compute MATV  
![](screenshots/27.png?raw=true "")  
3. Files are stored at the reported location  
![](screenshots/28.png?raw=true "")

### Optional functionalities
1. **Remove avoidance regions**  
   a. Load avoidance regions as segmentations  
      ![](screenshots/30.png?raw=true "") ![](screenshots/29.png?raw=true "")  
   b. Provide the 'avoidance segmentations' node name in the input bar  
      ![](screenshots/31.png?raw=true "")  
   c. The avoidance regions will be removed from the segmentation result  
2. **Remove area outside bounding boxes**  
   Selecting this option will result in the deletion of all areas outside the provided ROIs  
   ![](screenshots/32.png?raw=true "")  

## 3. Notes
### Segmentation
- Make sure the PET series name contains 'pet' or 'PET'  
![](screenshots/2.png?raw=true "")  
- Make sure the brain contour is named 'brain'  
![](screenshots/7.png?raw=true "")  
- Make sure the 'liverSphere' is present for liver-based segmentation methods  
![](screenshots/26.png?raw=true "")  
- Make sure only ***one*** patient is loaded into the scene  
### MATV extraction
- Make sure all segmentation results of the selected methods are present in the scene


## 4. License
See LICENSE

## 5. Contact
For any inquiries according to usage or bugs, please contact Kylie Keijzer ([k.keijzer@umcg.nl](mailto:k.keijzer@umcg.nl?subject=MUST-segmenter))

