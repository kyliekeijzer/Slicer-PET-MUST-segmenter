# MUST-segmenter
***MU***ltiple ***S***UV ***T***hresholding (MUST)-segmenter is a semi-automated PET image segmentation tool that enables delineation of 
multiple lesions at once, and extracts metabolic active tumor volume (MATV).

![](screenshots/Slicer-MUST-segmenter_screenshot.png?raw=true "MUST-segmenter screenshot")  

## 1. Installation
1. Download 3D Slicer as described [here](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#installing-3d-slicer).  
2. Download the '**SlicerRadiomics**' extension from the Extensions Manager as described [here](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html#install-extensions).

## 2. Tutorial

#### Sample Data
If needed, a sample dataset can be found [here](https://github.com/kyliekeijzer/Slicer-PET-MUST-segmenter/tree/master/Sample%20Data).

### A. Preparation

1. [Import PET and (optional) CT DICOM series into 3D Slicer ](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html#dicom-import)  
This can be retrieved by drag and dropping files into the DICOM database or by importing files into the DICOM database.
   - (optional) Import avoidance regions and brain contours as ***[Segmentations](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmentations.html#import-an-existing-segmentation-from-volume-file)***.

2. [Load patient data into the scene](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html#dicom-loading)  
Load ***ONE*** patient's PET (and CT) images (with or without avoidance regions and brain contours).  
![](screenshots/1.png?raw=true "")  
3. Display PET as overlay on CT
![](screenshots/3.png?raw=true "")  
4. Edit Window/Level display of PET and CT images and set to preferred settings  
a. ![](screenshots/4.png?raw=true "")  
b. ![](screenshots/5.png?raw=true "")  
5. Go to MUST-segmenter  
![](screenshots/6.png?raw=true "")  

### B. VOI characteristics
![](screenshots/43.png?raw=true "")  
1. Create a new Point List and rename to 'VOI'  
   ![](screenshots/8.png?raw=true "") &rarr; ![](screenshots/40.png?raw=true "") &rarr; ![](screenshots/16.png?raw=true "") &rarr; ![](screenshots/42.png?raw=true "")  
2. Place one seed in the desired location on the PET images  
3. Select a VOI diameter size  
4. 'Create VOI'  
5. Rename the VOI to an appropriate name  
   ![](screenshots/44.png?raw=true "")  
6. Repeat step 2 - 5 to create multiple VOIs  
7. 'Extract VOIs metrics' to save characteristics from all the VOIs

### C. Segmentation with MUST-segmenter
1. Identify tumors by seed placing (Points)  
   ![](screenshots/8.png?raw=true "") &rarr; ![](screenshots/40.png?raw=true "") &rarr; ![](screenshots/9.png?raw=true "")  
2. Choose segmentation method(s):  
   ![](screenshots/10.png?raw=true "")  
   - Fixed thresholds: **SUV 2.5**, **SUV 3.0** and **SUV 4.0**  
   - **41% SUVmax**  
     a. Place regions of interest (ROIs) around the identified tumors (the seeds don't need to be removed)  
        ![](screenshots/11.png?raw=true "") &rarr; ![](screenshots/41.png?raw=true "") &rarr;  
     (click on an image slice to initialize the annotation ROI)  
        ![](screenshots/12.png?raw=true "")  
     b. (optional) Take a threshold for each individual ROI  
        ![](screenshots/13.png?raw=true "")  
   - **A50P**  
     a. Create a VOI in the right lobe of the liver and in the right lung, as described in section A.  
     b. Rename the VOIs to 'VOI_liver' and 'VOI_lung'  
     c. *Delete* the 'VOI' Point List before segmentation!  
   - Liver-based thresholds: **Liver SUVmax** and **PERCIST**  
     a. Create a VOI in the right lobe of the liver as described in section A.  
     b. Rename the VOI to 'VOI_liver'  
     c. *Delete* the 'VOI' Point List before segmentation!  
   - Majority Voting: **MV2** and **MV3**  
     a. Make sure the following segmentation methods are *additionally* selected:  
        *SUV 2.5, SUV 4.0, 41% SUVmax, Liver SUVmax and PERCIST*  
   - **Brain region based**  
     a. Provide or create a brain segmentation  
        ![](screenshots/22.png?raw=true "")  
     b. Rename the brain segmentation node to 'brain'  
        ![](screenshots/23.png?raw=true "") &rarr; ![](screenshots/7.png?raw=true "")  
3. Perform segmentation and wait for the results!  
   ![](screenshots/24.png?raw=true "") &rarr; ![](screenshots/25.png?raw=true "")

### D. Feature extraction
1. Select the segmentation results that are present in the scene  
![](screenshots/10.png?raw=true "")  
2. Compute MATV  or Extract PET Features  
![](screenshots/27.png?raw=true "")  
3. Files are stored at the reported location  
![](screenshots/28.png?raw=true "")

### E. Optional functionalities
1. **Remove avoidance regions from segmentation result**  
   a. Load avoidance regions as segmentations  
      ![](screenshots/30.png?raw=true "") ![](screenshots/29.png?raw=true "")  
   b. Provide the 'avoidance segmentations' node name in the input bar  
      ![](screenshots/31.png?raw=true "")  
   c. The avoidance regions will be removed from the segmentation result  
2. **Remove area outside bounding boxes**  
   Selecting this option will result in the deletion of all areas outside the provided ROIs  
   ![](screenshots/32.png?raw=true "")  
3. **Reverse images**  
   Selecting this option will reverse the PET images order (may be needed if VOI metrics are inaccurate)  
4. **Edit VOIs positions**  
   If you are not satisfied with the position of the VOIs, they can be moved by the following steps:  
   a. Create and edit a new transform  
      ![](screenshots/36.png?raw=true "") &rarr; ![](screenshots/37.png?raw=true "")  
   b. Change the position by altering the 'Translation' and 'Rotation' values  
      ![](screenshots/38.png?raw=true "")  
   c. Permanently save the position changes  
      ![](screenshots/39.png?raw=true "")
5. **Save segmentations in the desired format**  
   Before exporting to NRRD or NIFTI, the segmentation node needs to be converted to a labelmap node!  
   ![](screenshots/35.png?raw=true "") &rarr;  
   ![](screenshots/33.png?raw=true "") &rarr; ![](screenshots/34.png?raw=true "")  
   

## 3. Notes
### Segmentation
- Make sure the PET series name contains 'pet' or 'PET'  
![](screenshots/2.png?raw=true "")  
- Make sure the brain contour is named 'brain'  
![](screenshots/7.png?raw=true "")  
- Make sure the '**VOI_liver**' is present for liver-based segmentation methods
- Make sure the '**VOI_liver**' and '**VOI_lung**' are present for the A50P segmentation method  
- Make sure only ***one*** patient is loaded into the scene  
### Feature extraction
- Make sure all segmentation results of the selected methods are present in the scene

## 4. SelfTest
A SelfTest (MUSTSegmenter) is available under the SelfTests and utilizes the Sample Data.

## 5. License
See LICENSE

## 6. Contact
For any inquiries according to usage or bugs, please contact Kylie Keijzer ([k.keijzer@umcg.nl](mailto:k.keijzer@umcg.nl?subject=MUST-segmenter))

