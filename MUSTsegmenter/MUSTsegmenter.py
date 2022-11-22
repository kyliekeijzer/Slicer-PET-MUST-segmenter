import os, datetime, traceback, pydicom, qt, slicer, vtk, sitkUtils
import numpy as np
import SimpleITK as sitk
import vtkSlicerSegmentationsModuleLogicPython as segmentLogic
from DICOMLib import DICOMUtils
from collections import deque
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# MUSTsegmenter
#


class MUSTsegmenter(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "MUST-segmenter"
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Kylie Keijzer (University Medical Center Groningen, The Netherlands)"]
    self.parent.helpText = """
    Please refer to https://github.com/kyliekeijzer/Slicer-MUST-segmenter
    """
    self.parent.acknowledgementText = """
    This segmentation extension was developed by Kylie Keijzer, 
    University Medical Center Groningen (UMCG), The Netherlands.
    """

#
# MUSTsegmenterWidget
#


class MUSTsegmenterWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.segmentationLogic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/MUSTsegmenter.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # set scene in MRML widgets
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # create logic classes
    self.segmentationLogic = MUSTsegmenterLogic()

    # App icon
    self.ui.icon.setPixmap(qt.QPixmap(self.resourcePath('Icons/MUSTsegmenter_small.png')))

    # Connections
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # Buttons
    self.ui.performSegmentationButton.connect('clicked(bool)', self.onSegmentationButton)
    self.ui.computeMATVButton.connect('clicked(bool)', self.onComputeMatvButton)
    self.ui.liverSphereButton.connect('clicked(bool)', self.onLiverSphereButton)

    self.initializeParameterNode()

  def cleanup(self):
    self.removeObservers()

  def enter(self):
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    # Do not react to parameter node changes
    self.removeObserver(self._parameterNode,
                        vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    # Parameter node will be reset
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    # If this module is shown while the scene is closed then
    # recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    self.setParameterNode(self.segmentationLogic.getParameterNode())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    """
    # Unobserve previously selected parameter node and add an observer to the newly selected
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode,
                          vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode,
                       vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    if self._parameterNode is None:
      return
    self.ui.performSegmentationButton.enabled = True

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    self._parameterNode.EndModify(wasModified)

  def checkValidParameters(self):
    """
    Method that checks if the user has provided valid segmentation parameters
    """
    # check for organ segmentations
    organSegmentsName = self.ui.organSegmentsName.plainText
    organSegments = None
    if organSegmentsName != '':
      try:
        organSegments = slicer.util.getNode(organSegmentsName)
      except:
        slicer.util.errorDisplay('No organ segmentations list found with name "' +
                                 organSegmentsName +
                                 '", please provide a valid list available in the scene.')
        return False
    self.organSegments = organSegments

    self.suvPerRoi = False
    # get selected segmentation methods
    thresholds = []
    if self.ui.SUV2_5.checkState() > 0:
      thresholds.append('suv2.5')
    if self.ui.SUV3_0.checkState() > 0:
      thresholds.append('suv3.0')
    if self.ui.SUV4_0.checkState() > 0:
      thresholds.append('suv4.0')
    if self.ui.SUV41max.checkState() > 0:
      if self.ui.suv_per_roi.checkState() > 0:
        self.suvPerRoi = True
      thresholds.append('41suvMax')
    if self.ui.LiverSUVmax.checkState() > 0:
      thresholds.append('liverSUVmax')
    if self.ui.PERCIST.checkState() > 0:
      thresholds.append('PERCIST')
    if self.ui.MV2.checkState() > 0:
      thresholds.append('MV2')
    if self.ui.MV3.checkState() > 0:
      thresholds.append('MV3')
    if self.ui.brain_region.checkState() > 0:
      try:
        slicer.util.getNode('brain')
        thresholds.append('brain_region')
      except:
        slicer.util.errorDisplay('No brain segmentation loaded, please load a segmentation named "brain" to use the '
                                 'brain region based thresholding method.')
        return False
    if len(thresholds) < 1:
      slicer.util.errorDisplay('No segmentation method selected, please select at least one segmentation method.')
      return False

    liverConditions = ['liverSUVmax' in thresholds, 'PERCIST' in thresholds]
    if any(liverConditions):
      try:
        slicer.util.getNode('liverSphere')
      except:
        slicer.util.errorDisplay('No sphere in liver, please create a liver sphere when using segmentation method '
                                 'Liver SUVmax or PERCIST')
        return False

    self.roiFilter = False
    if self.ui.ROIfilter.checkState() > 0:
      self.roiFilter = True

    self.segmentationColors = {
      'suv2.5': [0.12, 0.55, 0.18],
      'suv3.0': [0.0, 0.8, 0.2],
      'suv4.0': [0.55, 0.82, 0.35],
      '41suvMax': [0.22, 0.08, 0.94],
      'liverSUVmax': [0.08, 0.37, 0.94],
      'PERCIST': [0.04, 0.60, 0.87],
      'brainSuvMean50': [0.65, 0.82, 0.94],
      'brainSuvMean45': [0.65, 0.82, 0.94],
      'brainSuvMean41': [0.65, 0.82, 0.94],
      'brainSuvMeanCorrected50': [0.65, 0.82, 0.94],
      'brainSuvMeanCorrected45': [0.65, 0.82, 0.94],
      'brainSuvMeanCorrected41': [0.65, 0.82, 0.94],
      'MV2': [0.65, 0.0, 0.0],
      'MV3': [1.0, 0.48, 0.48]
    }

    self.segmentationMethods = thresholds
    return True

  def onLiverSphereButton(self):
    self.segmentationLogic.createSphere('liver')

  def onSegmentationButton(self):
    if self.checkValidParameters():
      self.segmentationLogic.performSegmentation(self.organSegments, self.segmentationMethods, self.suvPerRoi,
                                                 self.roiFilter, self.segmentationColors)

  def onComputeMatvButton(self):
    thresholds = []
    if self.ui.SUV2_5.checkState() > 0:
      thresholds.append('suv2.5')
    if self.ui.SUV3_0.checkState() > 0:
      thresholds.append('suv3.0')
    if self.ui.SUV4_0.checkState() > 0:
      thresholds.append('suv4.0')
    if self.ui.SUV41max.checkState() > 0:
      thresholds.append('41suvMax')
    if self.ui.LiverSUVmax.checkState() > 0:
      thresholds.append('liverSUVmax')
    if self.ui.PERCIST.checkState() > 0:
      thresholds.append('PERCIST')
    if self.ui.MV2.checkState() > 0:
      thresholds.append('MV2')
    if self.ui.MV3.checkState() > 0:
      thresholds.append('MV3')
    if self.ui.brain_region.checkState() > 0:
      thresholds.extend(['brainSuvMean41', 'brainSuvMean45', 'brainSuvMean50',
                         'brainSuvMeanCorrected41', 'brainSuvMeanCorrected45', 'brainSuvMeanCorrected50'])
    self.segmentationLogic.calulateMATV(thresholds)


#
# MUSTsegmenterLogic
#


class MUSTsegmenterLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.segmentationPerformed = False
    self.checkRequirements()

  def checkRequirements(self):
    try:
      import openpyxl
    except ModuleNotFoundError:
      if slicer.util.confirmOkCancelDisplay("MUST-segmenter requires the 'openpyxl' Python package. "
                                            "Click OK to install it now."):
        slicer.util.pip_install('openpyxl')
        import openpyxl

    try:
      import pandas as pd
      self.pd = pd
    except ModuleNotFoundError:
      if slicer.util.confirmOkCancelDisplay("MUST-segmenter requires the 'pandas' Python package. "
                                            "Click OK to install it now."):
        slicer.util.pip_install('pandas')
        import pandas as pd
        self.pd = pd

    try:
      from radiomics import featureextractor
      self.featureextractor = featureextractor
    except ModuleNotFoundError:
      slicer.util.errorDisplay("MUST-segmenter requires the 'SlicerRadiomics' extension, please download it in the "
                               "Extensions Manager.",
                               "SlicerRadiomics required")

  def convertNodesToSegmentationNode(self, originalNodes, fromLabelMap, fromStorage, method, thresholdDescr):
    """
    Method that converts the given list of vtkMRML nodes to a segmentation node.
    List must consist of either vtkMRMLModelNodes or vtkMRMLLabelMapVolumeNodes
    """
    # create segmentations node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')

    if method == 'liver':
      colors = [0.24, 0.42, 0.86]
      segmentationNode.SetName('liverSphere')
    else:
      colors = self.segmentationColors[thresholdDescr]
      segmentationNode.SetName('{0}_segmentation_{1}'.format(self.patientID, thresholdDescr))
    segmentations = segmentationNode.GetSegmentation()

    # convert original node to segmentation
    for i, node in enumerate(originalNodes):
      if fromLabelMap:
        # from labelmap
        vtkSegment = segmentLogic.vtkSlicerSegmentationsModuleLogic. \
          CreateSegmentFromLabelmapVolumeNode(node, segmentationNode)
      else:
        # from model
        vtkSegment = segmentLogic.vtkSlicerSegmentationsModuleLogic.CreateSegmentFromModelNode(node)
      if fromStorage:
        vtkSegment.SetName(node.GetName())
      else:
        vtkSegment.SetName('segment_{0}'.format(i))
      slicer.mrmlScene.RemoveNode(node)
      vtkSegment.SetColor(colors[0], colors[1], colors[2])
      segmentations.AddSegment(vtkSegment)

    # apply matrix transform to position the segmentation result
    if not fromLabelMap and method != 'liver':
      transformMatrix = vtk.vtkMatrix4x4()
      transformMatrix.SetElement(0, 0, -1.0)
      transformMatrix.SetElement(1, 1, -1.0)
      segmentationNode.ApplyTransformMatrix(transformMatrix)

    # create display node for segmentation result
    displayNode = slicer.vtkMRMLSegmentationDisplayNode()
    displayNode.SetOpacity2DFill(0.35)
    displayNode.SetSliceIntersectionThickness(4)
    slicer.mrmlScene.AddNode(displayNode)
    segmentationNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    segmentationNode.CreateClosedSurfaceRepresentation()
    segmentationNode.SetMasterRepresentationToClosedSurface()

  def getVolumeFromList(self, volumeName):
    """
    Method that searches the given volume name in all the visible volumes in the scene
    """
    volumes = slicer.mrmlScene.GetNodesByClass("vtkMRMLVolumeNode")
    selectedVolume = False
    for volume in volumes:
      fileName = volume.GetName()
      if volumeName.lower() in fileName.lower():
        selectedVolume = volume
        break
    return selectedVolume

  def getLiverSphereSuvValues(self, refVolume, suvImageArray):
    """
    Method that retrieves all the SUVs inside the liverSphere
    """
    sphere = slicer.util.getNode('liverSphere')
    sphere.SetReferenceImageGeometryParameterFromVolumeNode(refVolume)
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(sphere, labelmapVolumeNode,
                                                                             refVolume)
    sphereMask = slicer.util.arrayFromVolume(labelmapVolumeNode)
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    liverSphereValues = suvImageArray[sphereMask > 0.0]

    return liverSphereValues

  def createSphere(self, method):
    """
    Method that creates a sphere model node, for the given method
    """
    sphere = vtk.vtkSphereSource()
    # Initial positioning of the sphere from the fiducial points
    centerPointCoord = [0.0, 0.0, 0.0]

    if method == 'liver':
      radius = 15
      sphereName = 'liverSphere'
      try:
        liverMarkups = slicer.util.getNode(method)
        liverMarkups.GetNthFiducialPosition(0, centerPointCoord)
      except:
        slicer.util.errorDisplay("No seed found with name or ID 'liver'")
        return
    else:
      radius = 10
      sphereName = 'sphere'

    sphere.SetCenter(centerPointCoord)
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(30)
    sphere.SetThetaResolution(30)
    sphere.Update()

    # Create model node and add to scene
    model = slicer.vtkMRMLModelNode()
    model.SetName(sphereName)
    model.SetAndObservePolyData(sphere.GetOutput())
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetSliceIntersectionVisibility(True)
    slicer.mrmlScene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    slicer.mrmlScene.AddNode(model)
    self.convertNodesToSegmentationNode([model], False, True, method, method)

  def performSegmentation(self, organSegmentsNode, segmentationMethods, suvPerRoi, roiFilter, segmentationColors):
    """
    Method that performs the actual segmentation
    """
    self.segmentationColors = segmentationColors
    petSeriesName = 'pet'
    message = 'Segmentation finished'
    # pet dicom series
    petVolume = self.getVolumeFromList(petSeriesName)
    if not petVolume:
      slicer.util.errorDisplay(f"PET series not found, please add '{petSeriesName}' to the PET folder name and "
                               f"load PET series before performing segmentation.")
      return

    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    reversed = False
    petImageFileList = self.getPetFilesLists(petVolume, reversed)

    # get the organ segmentations
    try:
      organSegmentations = self.getArrayFromSegmentationNode(petVolume, organSegmentsNode)
    except:
      organSegmentations = None

    # seeds
    seeds = self.getPaintedSeedPoints(petVolume)

    # convert PET to SUV map
    suvMap, isEstimated = self.computeSuvMap(petImageFileList)
    shape = suvMap.shape
    seedSuvThresholds = self.computeSeedSuvThresholds(suvMap, petVolume, segmentationMethods)

    # set all organ values to 0.0 to filter out organs
    if organSegmentations is not None:
      suvMap[organSegmentations == 1.0] = 0.0

    if roiFilter:
      # ROIs
      ROIs = slicer.mrmlScene.GetNodesByClass("vtkMRMLAnnotationROINode")
      roisCoords = self.getRoisCoords(petVolume, ROIs)
      roisFilter = self.getRoisFilter(roisCoords, shape)
      suvMap[roisFilter == 0.0] = 0.0
    self.suvMap = suvMap
    self.petVolume = petVolume

    origin = petVolume.GetOrigin()
    spacing = petVolume.GetSpacing()

    mvMethods = []
    if 'MV2' in segmentationMethods or 'MV3' in segmentationMethods:
      if 'MV2' in segmentationMethods:
        segmentationsForMajVoting = []
        mvMethods.append('MV2')
      if 'MV3' in segmentationMethods:
        segmentationsForMajVoting = []
        mvMethods.append('MV3')
    else:
      segmentationsForMajVoting = None
    performMVsegmentation = segmentationsForMajVoting is not None

    for desc, threshold in seedSuvThresholds.items():
      # perform seed segmentation
      self.createSeedGrowingSegmentation(suvMap, seeds, threshold, desc, origin, spacing)
    # perform 41% SUVmax segmentation
    if '41suvMax' in segmentationMethods:
      self.createSuvMaxSegmentation(suvMap, suvPerRoi, shape, origin, spacing, petVolume, roiFilter)
    # perform majority vote segmentation
    if performMVsegmentation:
      majorityVotingMethods = ['suv2.5', 'suv4.0', '41suvMax', 'liverSUVmax', 'PERCIST']
      if set(majorityVotingMethods) <= set(segmentationMethods):
        self.createMajVotingSegmentation(segmentationsForMajVoting, mvMethods, origin, spacing, majorityVotingMethods,
                                         petVolume)
      else:
        message = f'Segmentation finished, but Majority Voting segmentation not performed. ' \
                  f'Please select the following methods to perform MV segmentation: ' \
                  f'SUV 2.5, SUV 4.0, 41% SUVmax, Liver SUVmax and PERCIST'

    qt.QApplication.setOverrideCursor(qt.Qt.ArrowCursor)
    self.segmentationPerformed = True
    slicer.util.infoDisplay(message, 'Segmentations created')

  def calulateMATV(self, thresholds):
    """
    Method that calculates the MATVs for the given threshold methods that are available in the scene
    """
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    self.matvRows = []
    extractor = self.featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(shape=['MeshVolume'])
    pixelVolume, pixelSpacing = self.getCubicCmPerPixel()
    suvImage = sitk.GetImageFromArray(self.suvMap)
    suvImage.SetSpacing(pixelSpacing)
    for thresh in thresholds:
      try:
        segmentNode = slicer.util.getNode('{0}_segmentation_{1}'.format(self.patientID, thresh))
      except:
        continue
      segmentArray = self.getArrayFromSegmentationNode(self.petVolume, segmentNode)
      matv = self.getMATV(extractor, segmentArray, pixelVolume, suvImage, pixelSpacing)
      self.matvRows.append({
        'Segmentation method': thresh,
        'Voxel Volume': matv[0],
        'Mesh Volume': matv[1]
      })
    volumeDf = self.pd.DataFrame(self.matvRows)
    savePath = "/".join(self.petSeriesPath.split('/')[:-1])
    volumeFilePath = f'{savePath}/MATV_patient_{self.patientID}.xlsx'
    volumeDf.to_excel(volumeFilePath, index=False)
    qt.QApplication.setOverrideCursor(qt.Qt.ArrowCursor)
    slicer.util.infoDisplay(f'MATV calculations finished, MATVs stored at: {volumeFilePath}', 'MATVs extracted')

  def getRoisFilter(self, petRoisInfo, shape):
    """
    Method that creates a general ROI mask
    """
    roisNumber = len(petRoisInfo)
    roisFilter = np.zeros(shape)
    for i in range(0, roisNumber):
      roiInfo = petRoisInfo[i]
      center = np.array(roiInfo['center'])
      radius = np.abs(roiInfo['radius'])
      top = center - radius
      bottom = center + radius
      roisFilter[top[0]:bottom[0], top[1]:bottom[1], top[2]:bottom[2]] = 1.0

    return roisFilter

  def createSeedGrowingSegmentation(self, suvMap, seeds, suvThreshold, thresholdDescr, origin, spacing):
    """
    Method that handles the creation of the segmentation result.
    """
    labelmapNodes = []
    shape = suvMap.shape
    self.maxX, self.maxY, self.maxZ = shape[2], shape[1], shape[0]
    self.suvImageArray = suvMap.copy()
    self.suvThreshold = suvThreshold
    self.thresholdDescr = thresholdDescr
    for i, seed in enumerate(seeds):
      # perform the segmentation for the given seed
      segmentImage, segmentArray = self.seedGrowSegmentation(seed)
      # save the segmentation binary array
      if np.count_nonzero(segmentArray) > 1:
        # set created segment to zero for next seed
        self.suvImageArray[segmentArray == 1] = 0.0
        # convert segmentation image to a labelmap node
        labelmapNode = self.createLabelMapNode(origin, segmentImage, spacing)
        labelmapNodes.append(labelmapNode)

    self.convertNodesToSegmentationNode(labelmapNodes, True, False, False, thresholdDescr)

  def createSuvMaxSegmentation(self, suvMap, suvPerRoi, shape, origin, spacing, petVolume, roiFilter):
    """
    Method that creates the segmentation result based on SUVmax thresholding
    """
    ROIs = slicer.mrmlScene.GetNodesByClass("vtkMRMLAnnotationROINode")
    roisCoords = self.getRoisCoords(petVolume, ROIs)

    if suvPerRoi:
      labelmapNodes = []
      for i, roi in roisCoords.items():
        suvMapCopy = suvMap.copy()
        roiArray = self.getRoisFilter({0: roi}, shape)
        suvMapCopy[roiArray == 0.0] = 0.0
        suvMaxThresh = np.max(suvMapCopy) * 0.41
        segmentMask = np.zeros(shape)
        segmentMask[suvMapCopy >= suvMaxThresh] = 1.0
        if np.count_nonzero(segmentMask) > 1:
          segmentImage = sitk.GetImageFromArray(segmentMask)
          labelmapNode = self.createLabelMapNode(origin, segmentImage, spacing)
          labelmapNodes.append(labelmapNode)
      self.convertNodesToSegmentationNode(labelmapNodes, True, False, False, '41suvMax')

    else:
      if not roiFilter:
        roisFilter = self.getRoisFilter(roisCoords, shape)
        suvMap[roisFilter == 0.0] = 0.0
      suvMaxThresh = np.max(suvMap) * 0.41
      segmentMask = np.zeros(shape)
      segmentMask[suvMap >= suvMaxThresh] = 1.0
      if np.count_nonzero(segmentMask) > 1:
        segmentImage = sitk.GetImageFromArray(segmentMask)
        labelMapNode = self.createLabelMapNode(origin, segmentImage, spacing)
        self.convertNodesToSegmentationNode([labelMapNode], True, False, False, '41suvMax')

  def createMajVotingSegmentation(self, segmentationsForMajVoting, mvMethods, origin, spacing, segmentationMethods,
                                  petVolume):
    """
    Method that creates the segmentation result based on majority voting
    """
    for method in segmentationMethods:
      segmentNode = slicer.util.getNode('{0}_segmentation_{1}'.format(self.patientID, method))
      segmentArray = self.getArrayFromSegmentationNode(petVolume, segmentNode)
      segmentationsForMajVoting.append(segmentArray)
    voxelOverlaps = sum(segmentationsForMajVoting)
    if not isinstance(voxelOverlaps, int):
      for method in mvMethods:
        if method == 'MV2':
          thresh = 2.0
        else:
          thresh = 3.0
        mv = voxelOverlaps.copy()
        mv = np.where(mv >= thresh, 1.0, 0.0)
        if np.count_nonzero(mv) > 1:
          segmentImage = sitk.GetImageFromArray(mv)
          labelMapNode = self.createLabelMapNode(origin, segmentImage, spacing)
          self.convertNodesToSegmentationNode([labelMapNode], True, False, False, method)

  def createLabelMapNode(self, origin, segmentImage, spacing):
    """
    Method that converts an image to a LabelMapVolumeNode
    """
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    sitk.WriteImage(segmentImage, sitkUtils.GetSlicerITKReadWriteAddress(labelmapNode.GetName()))
    labelmapNode.GetImageData().Modified()
    labelmapNode.Modified()
    labelmapNode.SetOrigin(origin)
    labelmapNode.SetSpacing(spacing)
    return labelmapNode

  def getMATV(self, extractor, segmentArray, pixelVolume, suvImage, pixelSpacing):
    """
    Method that calculates the MATV for the given segmentations
    """
    pixelAmount = np.count_nonzero(segmentArray)
    voxelMatv = pixelAmount * pixelVolume
    meshMatv = self.getPyRadiomicsMatv(extractor, segmentArray, suvImage, pixelSpacing)

    return [voxelMatv, meshMatv]

  def getPyRadiomicsMatv(self, extractor, segmentArray, suvImage, pixelSpacing):
    """
    Method that calculates the tumor volume based on triangle mesh using PyRadiomics
    """
    segmentImage = sitk.GetImageFromArray(segmentArray)
    segmentImage.SetSpacing(pixelSpacing)
    featureVector = extractor.execute(suvImage, segmentImage)
    matv = featureVector['original_shape_MeshVolume'] / 1000  # in cc

    return matv

  def getCubicCmPerPixel(self):
    """
    Method that computes the cubic cm per pixel for the given PET DICOM series
    """
    if not self.segmentationPerformed:
      self.setPetDataParameters()
      self.segmentationPerformed = True
    spacingX, spacingY, sliceThickness = self.getPixelSpacing()
    pixelVolume = spacingX * spacingY * sliceThickness / 1000  # cc
    return pixelVolume, [spacingX, spacingY, sliceThickness]

  def getPixelSpacing(self):
    """
    Method that retrieves the pixel spacing for the given PET DICOM series
    """
    pixel_spacing = self.PixelSpacing
    slice_thickness = self.SliceThickness

    return pixel_spacing[0], pixel_spacing[1], slice_thickness

  def setPetDataParameters(self):
    self.petVolume = self.getVolumeFromList('pet')
    petImageFileList = self.getPetFilesLists(self.petVolume, False)
    self.suvMap, isEstimated = self.computeSuvMap(petImageFileList)

  def getBrainSuvArray(self, suvImageArray, petVolume):
    """
    Method that retrieves the suv values array located at the brain
    """
    brainSegment = slicer.util.getNode("brain")
    brainSegment.SetReferenceImageGeometryParameterFromVolumeNode(petVolume)
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(brainSegment, labelmapVolumeNode,
                                                                         slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
    brainArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
    brainSuvs = suvImageArray[brainArray > 0]
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    return brainSuvs

  def computeSeedSuvThresholds(self, suvImageArray, petVolume, segmentationMethods):
    """
    Method that computes the SUV thresholds that will be used for seed segmentation
    """
    liverConditions = ['liverSUVmax' in segmentationMethods, 'PERCIST' in segmentationMethods]
    if any(liverConditions):
      liverSuvValues = self.getLiverSphereSuvValues(petVolume, suvImageArray)

    thresholds = {}
    for method in segmentationMethods:
      if method == 'suv2.5':
        thresholds[method] = 2.5
      elif method == 'suv3.0':
        thresholds[method] = 3.0
      elif method == 'suv4.0':
        thresholds[method] = 4.0
      elif method == 'liverSUVmax':
        thresholds[method] = np.max(liverSuvValues)
      elif method == 'PERCIST':
        percistSuv = 1.5 * np.mean(liverSuvValues) + (2 * np.std(liverSuvValues))
        thresholds[method] = percistSuv
      elif method == 'brain_region':
        brainSuvArray = self.getBrainSuvArray(suvImageArray, petVolume)
        brainSuvMean = np.mean(brainSuvArray)
        brainSuvMean50 = 0.5 * brainSuvMean
        brainSuvMean45 = 0.45 * brainSuvMean
        brainSuvMean41 = 0.41 * brainSuvMean

        thresholds["brainSuvMean50"] = brainSuvMean50
        thresholds["brainSuvMean45"] = brainSuvMean45
        thresholds["brainSuvMean41"] = brainSuvMean41

        # Age correction: (age - 20) * 0.125 / (70 - 20)
        ageCorrection = (self.patientAge - 20) * (0.125 / 50)
        brainSuvMeanCorrected = brainSuvMean + ageCorrection
        brainSuvMeanCorrected50 = 0.5 * brainSuvMeanCorrected
        brainSuvMeanCorrected45 = 0.45 * brainSuvMeanCorrected
        brainSuvMeanCorrected41 = 0.41 * brainSuvMeanCorrected

        thresholds["brainSuvMeanCorrected50"] = brainSuvMeanCorrected50
        thresholds["brainSuvMeanCorrected45"] = brainSuvMeanCorrected45
        thresholds["brainSuvMeanCorrected41"] = brainSuvMeanCorrected41

    return thresholds

  def seedGrowSegmentation(self, seed):
    """
    Method that performs the SUV-based segmentation starting from the given seed point and
    based on 6-connected neighbouring pixels.
    """
    # create queue with neighbours to check
    self.neighbourQueue = deque()
    self.neighbourQueue.append((seed[0], seed[1], seed[2]))
    # create output segment
    self.outputSegment = np.zeros(self.suvImageArray.shape)
    self.outputSegment[seed[0], seed[1], seed[2]] = 1

    while len(self.neighbourQueue) != 0:
      newItem = self.neighbourQueue.pop()
      self.checkNeighbourSuv(newItem[0], newItem[1], newItem[2] - 1)
      self.checkNeighbourSuv(newItem[0], newItem[1], newItem[2] + 1)
      self.checkNeighbourSuv(newItem[0], newItem[1] - 1, newItem[2])
      self.checkNeighbourSuv(newItem[0], newItem[1] + 1, newItem[2])
      self.checkNeighbourSuv(newItem[0] - 1, newItem[1], newItem[2])
      self.checkNeighbourSuv(newItem[0] + 1, newItem[1], newItem[2])

    segmentImage = sitk.GetImageFromArray(self.outputSegment)
    return segmentImage, self.outputSegment

  def checkNeighbourSuv(self, z, y, x):
    if (x < self.maxX and y < self.maxY and z < self.maxZ and x > -1 and y > -1 and z > -1):
      suvValue = self.suvImageArray[z, y, x]
      if self.isSuvCriteriaAccepted(suvValue) and self.outputSegment[z, y, x] == 0:
        self.outputSegment[z, y, x] = 1
        self.neighbourQueue.appendleft((z, y, x))

  def isSuvCriteriaAccepted(self, suvValue):
    if suvValue >= self.suvThreshold:
      return True
    return False

  def getArrayFromSegmentationNode(self, volumeNode, segmentsNode):
    """
    Method that gets all the organ segmentations from the given node name
    """
    segmentsNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentsNode, labelmapVolumeNode,
                                                                         slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
    segmentationsArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
    # set segment pixels to 1.0
    segmentationsArray[segmentationsArray > 0] = 1.0
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    return segmentationsArray

  def readDicomSeriesFiles(self, imageFileList):
    """
    Method that reads an image series into a SimpleITK image.
    Slope/intercept rescaling is performed.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(imageFileList)
    image = reader.Execute()
    imageArray = sitk.GetArrayFromImage(image)

    return image, imageArray

  def computeSuvMap(self, imageFileList):
    """
    Method that computes the SUVmap of a raw input PET volume.
    It is assumed that the calibration factor was applied beforehand to the PET
    volume (e.g., rawPET = rawPET * RescaleSlope + RescaleIntercept).
    """
    isEstimated = False
    image, imageArray = self.readDicomSeriesFiles(imageFileList)
    suvMap = np.zeros(imageArray.shape)

    # retrieve general patient and scan information
    dicomSeries = pydicom.dcmread(imageFileList[0])
    # get patient and scan ifo
    self.patientID = dicomSeries.PatientID
    self.patientAge = int(dicomSeries.PatientAge[:-1])
    self.PixelSpacing = dicomSeries.PixelSpacing
    self.SliceThickness = dicomSeries.SliceThickness

    try:
      # get patient weight (grams)
      weight = float(dicomSeries.PatientWeight) * 1000
      # start time for the Radiopharmaceutical Injection
      injectionTime = datetime.datetime.strptime(
        dicomSeries.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime, '%H%M%S.%f')
      # half life for Radionuclide (seconds)
      halfLife = float(dicomSeries.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
      # total dose injected for Radionuclide (Becquerels Bq)
      injectedDose = float(dicomSeries.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    except:
      # make estimation
      traceback.print_exc()
      weight = 75000
      injectionTime = 3600
      halfLife = 6588
      injectedDose = 420000000
      isEstimated = True

    # iter through all slices to compute SUV values
    for index, slice in enumerate(imageFileList):
      dicomSeries = pydicom.dcmread(slice)
      try:
        # get scan time
        scantime = datetime.datetime.strptime(dicomSeries.AcquisitionTime, '%H%M%S.%f')
        # calculate decay
        decay = np.exp(-np.log(2) * ((scantime - injectionTime).seconds) / halfLife)
        # calculate the dose decayed during procedure (Bq)
        injectedDoseDecay = injectedDose * decay
      except:
        traceback.print_exc()
        decay = np.exp(-np.log(2) * (1.75 * 3600) / 6588)  # 90 min waiting time, 15 min preparation
        injectedDoseDecay = 420000000 * decay  # 420 MBq
        isEstimated = True

      # Calculate SUV (g/ml)
      suvMap[index, :, :] = imageArray[index, :, :] * weight / injectedDoseDecay

    return suvMap, isEstimated

  def getPetFilesLists(self, node, reversed):
    """
    Method that retrieves the PET image series paths
    """
    self.petSeriesPath = self.getScanFilesPath(node)
    petSeriesFiles = [os.path.join(self.petSeriesPath, x) for x in os.listdir(self.petSeriesPath)]
    petSeriesFiles = self.sortDicomByInstanceNum(petSeriesFiles, reversed)

    return petSeriesFiles

  def getScanFilesPath(self, node):
    """
    Method that retrieves the storage files path of the given node
    """
    storageNode = node.GetStorageNode()
    if storageNode is not None:  # loaded via drag-drop
      filepath = storageNode.GetFullNameFromFileName()
    else:  # loaded via DICOM browser
      instanceUIDs = node.GetAttribute('DICOM.instanceUIDs').split()
      filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])
    filepath = '/'.join(filepath.split('/')[:-1])
    return filepath

  def sortDicomByInstanceNum(self, imageFileList, reversed):
    """
    Method that sorts a list of images paths by their instance number
    """
    data = []
    for image in imageFileList:
      dicomFile = pydicom.dcmread(image)
      data.append({'dicomFile': image, 'n': dicomFile.InstanceNumber})
    data = sorted(data, key=lambda x: x['n'])
    if reversed:
      return [x['dicomFile'] for x in data][::-1]
    else:
      return [x['dicomFile'] for x in data]

  def getPaintedSeedPoints(self, refVolume):
    """
    Method that retrieves the user provided painted seeds
    """
    seedCoordinates = []
    fidLists = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode")
    for fidList in fidLists:
      numFids = fidList.GetNumberOfFiducials()
      for i in range(numFids):
        isVisible = fidList.GetNthFiducialVisibility(i)
        if isVisible:
          zxyCoords = self.getSeedCoordinate(fidList, i, refVolume)
          seedCoordinates.append(zxyCoords)

    return seedCoordinates

  def getSeedCoordinate(self, fidList, i, refVolume):
    """
    Method that retrieves the IJK coordinate from a given fiducial index in the provided fiducials list
    and reference volume.
    """
    # Get RAS coordinates of seed
    rasCoords = [0, 0, 0, 1]
    fidList.GetNthFiducialWorldCoordinates(i, rasCoords)
    # Convert RAS to ZXY
    zxyCoords = self.convertRasPointToIjkPoint(rasCoords[0:3], refVolume)

    return zxyCoords

  def convertRasPointToIjkPoint(self, rasCoords, refVolume):
    """
    Method that converts a RAS point to ZYX coordinate, given the reference volume
    """
    # If PET volume is transformed, apply that transform to get PET RAS coordinates
    transformRasToVolumeRas = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, refVolume.GetParentTransformNode(),
                                                         transformRasToVolumeRas)
    refVolumeRas = transformRasToVolumeRas.TransformPoint(rasCoords)
    # Get voxel coordinates from physical coordinates
    volumeRasToIjk = vtk.vtkMatrix4x4()
    refVolume.GetRASToIJKMatrix(volumeRasToIjk)
    ijkCoords = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(np.append(refVolumeRas, 1.0), ijkCoords)
    ijkCoords = tuple([int(round(c)) for c in ijkCoords[0:3]])
    # Flip ijk vector to zxy components
    zyxCoords = np.flip(ijkCoords).tolist()

    return zyxCoords

  def getRoisCoords(self, petVolume, ROIs):
    """
    Method that gets the coordinates of the given ROIs
    """
    roisCoords = {}
    for i, roi in enumerate(ROIs):
      center, radius = self.getRoiCoord(petVolume, roi)
      roisCoords[i] = {
        'center': center,
        'radius': radius
      }

    return roisCoords

  def getRoiCoord(self, petVolume, roi):
    """
    Method that extracts the center and the radius from the given ROI
    """
    roi.SetVolumeNodeID(petVolume.GetID())
    # Get RAS coordinates
    center = [0, 0, 0]
    radius = [0, 0, 0]
    roi.GetControlPointWorldCoordinates(0, center)
    roi.GetControlPointWorldCoordinates(1, radius)
    center = np.array(center)
    radius = np.array(radius)
    upper = np.add(center, radius)
    # z y x
    # slice, row, column
    center = self.convertRasPointToIjkPoint(center, petVolume)
    upper = np.array(self.convertRasPointToIjkPoint(upper, petVolume))
    radius = np.subtract(upper, center).tolist()

    return center, radius

#
# SelfTest for MUSTsegmenter
#


class MUSTsegmenterTest(ScriptedLoadableModuleTest):
  def setUp(self):
    slicer.mrmlScene.Clear(0)
    self.tempDataDir = "\\\\zkh\\dfs\\Gebruikers14\\KeijzerK\\Data\\Downloads"
    # self.tempDataDir = os.path.join(slicer.app.temporaryPath, 'MUSTSegmenterTest')
    self.tempDicomDatabaseDir = os.path.join(slicer.app.temporaryPath, 'MUSTSegmenterTestDicom')

    # segmentation parameters
    self.segmentationLogic = MUSTsegmenterLogic()
    self.organSegments = None
    self.suvPerRoi = False
    self.roiFilter = False
    self.segmentationColors = {
      'suv2.5': [0.12, 0.55, 0.18],
      'suv3.0': [0.0, 0.8, 0.2],
      'suv4.0': [0.55, 0.82, 0.35],
      '41suvMax': [0.22, 0.08, 0.94],
      'liverSUVmax': [0.08, 0.37, 0.94],
      'PERCIST': [0.04, 0.60, 0.87],
      'MV2': [0.65, 0.0, 0.0],
      'MV3': [1.0, 0.48, 0.48]
    }
    self.segmentationMethods = ['suv2.5', 'suv3.0', 'suv4.0',
                                '41suvMax', 'liverSUVmax', 'PERCIST',
                                'MV2', 'MV3']

  def runTest(self):
    self.setUp()
    self.testSegmentation()

  def testSegmentation(self):
    self.delayDisplay('Starting the test')
    self.loadTestData()
    self.performSegmentationTests()

  def performSegmentationTests(self):
    self.segmentationLogic.performSegmentation(self.organSegments, self.segmentationMethods, self.suvPerRoi,
                                               self.roiFilter, self.segmentationColors)
    self.segmentationLogic.calulateMATV(self.segmentationMethods)

  def loadTestData(self):
    zipUrl = "https://github.com/kyliekeijzer/Slicer-PET-MUST-segmenter/raw/master/Sample%20Data/Sample%20Data.zip"
    zipFilePath = self.tempDataDir + '\\dicom1.zip'
    zipFileData = self.tempDataDir + '\\dicom1'

    if not os.access(self.tempDataDir, os.F_OK):
      os.mkdir(self.tempDataDir)
    if not os.access(zipFileData, os.F_OK):
      os.mkdir(zipFileData)
      if not os.path.isfile(zipFilePath):
        slicer.util.downloadAndExtractArchive(zipUrl, zipFilePath, zipFileData)
      else:
        import zipfile
        with zipfile.ZipFile(zipFilePath, 'r') as zipFile:
          zipFile.extractall(zipFileData)
    DICOMUtils.importDicom(zipFileData + "\\Sample Data\\PET")

    # Load PET
    dicomFiles = slicer.util.getFilesInDirectory(zipFileData + "\\Sample Data\\PET")
    loadablesByPlugin, loadEnabled = DICOMUtils.getLoadablesFromFileLists([dicomFiles], ['DICOMScalarVolumePlugin'])
    loadedNodeIDs = DICOMUtils.loadLoadables(loadablesByPlugin)
    imageNode = slicer.mrmlScene.GetNodeByID(loadedNodeIDs[0])

    suvNormalizationFactor = 0.00040166400000000007
    quantity = slicer.vtkCodedEntry()
    quantity.SetFromString('CodeValue:126400|CodingSchemeDesignator:DCM|CodeMeaning:Standardized Uptake Value')
    units = slicer.vtkCodedEntry()
    units.SetFromString(
      'CodeValue:{SUVbw}g/ml|CodingSchemeDesignator:UCUM|CodeMeaning:Standardized Uptake Value body weight')
    multiplier = vtk.vtkImageMathematics()
    multiplier.SetOperationToMultiplyByK()
    multiplier.SetConstantK(suvNormalizationFactor)
    multiplier.SetInput1Data(imageNode.GetImageData())
    multiplier.Update()
    imageNode.GetImageData().DeepCopy(multiplier.GetOutput())
    imageNode.GetVolumeDisplayNode().SetWindowLevel(6, 3)
    imageNode.GetVolumeDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeInvertedGrey')
    imageNode.SetVoxelValueQuantity(quantity)
    imageNode.SetVoxelValueUnits(units)

    # load seeds and ROIs
    slicer.util.loadMarkups(zipFileData + "\\Sample Data\\seed.mrk.json")
    import json
    roiNames = ['R', 'R_1', 'R_2', 'R_3']
    for r in roiNames:
      rObj = open(zipFileData + "\\Sample Data\\" + r + ".json")
      coords = json.load(rObj)
      roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLAnnotationROINode")
      roi.SetName(r)
      roi.SetRadiusXYZ(list(coords['radius']))
      center = list(coords['center'])
      roi.SetXYZ(center[0], center[1], center[2])
    # load liverSphere
    liverSphere = slicer.util.loadSegmentation(zipFileData + "\\Sample Data\\liverSphere.seg.vtm")
    liverSphere.SetName('liverSphere')

