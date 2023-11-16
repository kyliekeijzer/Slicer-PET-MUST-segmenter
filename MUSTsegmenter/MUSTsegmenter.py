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
    Please refer to https://github.com/kyliekeijzer/Slicer-PET-MUST-segmenter
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
    iconPath = os.path.join(os.path.dirname(slicer.modules.mustsegmenter.path), 'Resources', 'Icons',
                            'MUSTsegmenter.png')
    self.ui.icon.setPixmap(qt.QPixmap(iconPath).scaled(250, 250))

    # Connections
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # Buttons
    self.ui.performSegmentationButton.connect('clicked(bool)', self.onSegmentationButton)
    self.ui.computeMATVButton.connect('clicked(bool)', self.onComputeMatvButton)
    self.ui.extractPETFeaturesButton.connect('clicked(bool)', self.onExtractPetFeaturesButton)
    self.ui.createVoiButton.connect('clicked(bool)', self.onCreateSphereButton)
    self.ui.extractVOIsMetricsButton.connect('clicked(bool)', self.onExtractVoiMetricsButton)

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
    if self.ui.SUV41maxSeed.checkState() > 0:
      thresholds.append('41suvMaxSeed')
    if self.ui.LiverSUVmax.checkState() > 0:
      thresholds.append('liverSUVmax')
    if self.ui.PERCIST.checkState() > 0:
      thresholds.append('PERCIST')
    if self.ui.A50P.checkState() > 0:
      try:
        slicer.util.getNode('VOI_liver')
        slicer.util.getNode('VOI_lung')
        thresholds.append('A50P')
      except:
        slicer.util.errorDisplay('No spheres in the liver and/or lung, please create these using the "VOI placement" '
                                 'functionality, since they are required for A50P.')
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
        slicer.util.getNode('VOI_liver')
      except:
        slicer.util.errorDisplay('No sphere in liver, please create a liver sphere when using segmentation method '
                                 'Liver SUVmax or PERCIST')
        return False

    self.roiFilter = False
    if self.ui.ROIfilter.checkState() > 0:
      self.roiFilter = True

    self.reversed = False
    if self.ui.reversed.checkState() > 0:
      self.reversed = True

    self.segmentationColors = {
      'suv2.5': [0.12, 0.55, 0.18],
      'suv3.0': [0.0, 0.8, 0.2],
      'suv4.0': [0.55, 0.82, 0.35],
      '41suvMax': [0.22, 0.08, 0.94],
      '41suvMaxSeed': [0.22, 0.08, 0.94],
      'liverSUVmax': [0.08, 0.37, 0.94],
      'PERCIST': [0.04, 0.60, 0.87],
      'A50P': [0.26, 0.16, 0.79],
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

  def onCreateSphereButton(self):
    diameter = int(self.ui.voiSize.value)
    self.segmentationLogic.createSphere(diameter, "VOI", False)

  def onExtractVoiMetricsButton(self):
    reversed = False
    if self.ui.reversed.checkState() > 0:
      reversed = True
    self.segmentationLogic.extractVOIsMetrics(reversed)

  def onSegmentationButton(self):
    if self.checkValidParameters():
      self.segmentationLogic.performSegmentation(self.organSegments, self.segmentationMethods, self.suvPerRoi,
                                                 self.roiFilter, self.reversed, self.segmentationColors)

  def onComputeMatvButton(self):
    reversed = False
    if self.ui.reversed.checkState() > 0:
      reversed = True
    thresholds = self.getSelectedThresholds()
    self.segmentationLogic.calulateMATV(thresholds, reversed)

  def onExtractPetFeaturesButton(self):
    reversed = False
    if self.ui.reversed.checkState() > 0:
      reversed = True
    thresholds = self.getSelectedThresholds()
    self.segmentationLogic.extractFeatures(thresholds, reversed)

  def getSelectedThresholds(self):
    thresholds = []
    if self.ui.SUV2_5.checkState() > 0:
      thresholds.append('suv2.5')
    if self.ui.SUV3_0.checkState() > 0:
      thresholds.append('suv3.0')
    if self.ui.SUV4_0.checkState() > 0:
      thresholds.append('suv4.0')
    if self.ui.SUV41max.checkState() > 0:
      thresholds.append('41suvMax')
    if self.ui.SUV41maxSeed.checkState() > 0:
      thresholds.append('41suvMaxSeed')
    if self.ui.LiverSUVmax.checkState() > 0:
      thresholds.append('liverSUVmax')
    if self.ui.PERCIST.checkState() > 0:
      thresholds.append('PERCIST')
    if self.ui.A50P.checkState() > 0:
      thresholds.append('A50P')
    if self.ui.MV2.checkState() > 0:
      thresholds.append('MV2')
    if self.ui.MV3.checkState() > 0:
      thresholds.append('MV3')
    if self.ui.brain_region.checkState() > 0:
      thresholds.extend(['brainSuvMean41', 'brainSuvMean45', 'brainSuvMean50',
                         'brainSuvMeanCorrected41', 'brainSuvMeanCorrected45', 'brainSuvMeanCorrected50'])
    return thresholds


#
# MUSTsegmenterLogic
#


class MUSTsegmenterLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
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
      from torch.nn import Conv3d
      from torch import tensor
      self.Conv3d = Conv3d
      self.torchTensor = tensor
    except ModuleNotFoundError:
      if slicer.util.confirmOkCancelDisplay("MUST-segmenter requires the 'PyTorch' Python package. "
                                            "Click OK to install it now."):
        slicer.util.pip_install('torch')
        from torch.nn import Conv3d
        from torch import tensor
        self.Conv3d = Conv3d
        self.torchTensor = tensor

    slicer.modules.slicerradiomics.widgetRepresentation()  # SlicerRadiomics installs additional python packages on widget instantiation
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

    if method == 'VOI' or method == 'peakSphere' or 'A50P' in method or '41suvMaxSeed' in method:
      colors = [0.24, 0.42, 0.86]
      thickness = 1
      segmentationNode.SetName(f'{method}_')
    else:
      colors = self.segmentationColors[thresholdDescr]
      thickness = 4
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
    if 'A50P' not in method and '41suvMaxSeed' not in method and not fromLabelMap and method != 'VOI':
      transformMatrix = vtk.vtkMatrix4x4()
      transformMatrix.SetElement(0, 0, -1.0)
      transformMatrix.SetElement(1, 1, -1.0)
      segmentationNode.ApplyTransformMatrix(transformMatrix)

    # create display node for segmentation result
    displayNode = slicer.vtkMRMLSegmentationDisplayNode()
    displayNode.SetOpacity2DFill(0.35)
    displayNode.SetSliceIntersectionThickness(thickness)
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

  def getSphereSuvValues(self, refVolume, suvImageArray, voiName):
    """
    Method that retrieves all the SUVs inside the given VOI
    """
    sphere = slicer.util.getNode(voiName)
    sphere.SetReferenceImageGeometryParameterFromVolumeNode(refVolume)
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(sphere, labelmapVolumeNode,
                                                                             refVolume)
    sphereMask = slicer.util.arrayFromVolume(labelmapVolumeNode)
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    sphereValues = suvImageArray[sphereMask > 0.0]

    return sphereValues

  def createSphere(self, diameter, nodeName, rasCoords):
    """
    Method that creates a sphere model node, for the given method
    """
    sphere = vtk.vtkSphereSource()
    radius = diameter * 10 / 2

    if rasCoords:
      centerPointCoord = rasCoords
    else:
      try:
        markups = slicer.util.getNode(nodeName)
        centerPointCoord = markups.GetNthControlPointPosition(0)
      except:
        slicer.util.errorDisplay(f"No seed found with name or ID '{nodeName}'")
        return

    sphere.SetCenter(centerPointCoord)
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(30)
    sphere.SetThetaResolution(30)
    sphere.Update()

    # Create model node and add to scene
    model = slicer.vtkMRMLModelNode()
    model.SetName(nodeName)
    model.SetAndObservePolyData(sphere.GetOutput())
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetSliceIntersectionVisibility(True)
    slicer.mrmlScene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    slicer.mrmlScene.AddNode(model)
    self.convertNodesToSegmentationNode([model], False, True, nodeName, nodeName)

  def performSegmentation(self, organSegmentsNode, segmentationMethods, suvPerRoi, roiFilter, reversed, segmentationColors):
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

    petImageFileList = self.getPetFilesLists(petVolume, reversed)

    # get the organ segmentations
    try:
      organSegmentations = self.getArrayFromSegmentationNode(petVolume, organSegmentsNode)
    except:
      organSegmentations = None

    # seeds
    seeds = self.getPaintedSeedPoints(petVolume, segmentationMethods)

    # convert PET to SUV map
    suvMap, isEstimated = self.computeSuvMap(petImageFileList)
    shape = suvMap.shape
    seedSuvThresholds = self.computeSeedSuvThresholds(suvMap, petVolume, segmentationMethods)

    # set all organ values to 0.0 to filter out organs
    if organSegmentations is not None:
      suvMap[organSegmentations == 1.0] = 0.0

    if roiFilter:
      # ROIs
      roisCoords = self.getRoisCoords(petVolume)
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
    if '41suvMaxSeed' in segmentationMethods:
      self.createSuvMaxSeedSegmentation(suvMap, seeds, shape, origin, spacing)
    # perform majority vote segmentation
    if performMVsegmentation:
      if '41suvMaxSeed' in segmentationMethods:
        maxMethod = '41suvMaxSeed'
      else:
        maxMethod = '41suvMax'
      majorityVotingMethods = ['suv2.5', 'suv4.0', maxMethod, 'liverSUVmax', 'PERCIST']
      finished = self.createMajVotingSegmentation(segmentationsForMajVoting, mvMethods, origin,
                                                  spacing, majorityVotingMethods, petVolume)
      if not finished:
        message = f'Segmentation finished, but Majority Voting segmentation not performed. ' \
                  f'Please select the following methods to perform MV segmentation: ' \
                  f'SUV 2.5, SUV 4.0, 41% SUVmax (ROI or seed-based), Liver SUVmax and PERCIST'

    qt.QApplication.setOverrideCursor(qt.Qt.ArrowCursor)
    slicer.util.infoDisplay(message, 'Segmentations created')

  def calulateMATV(self, thresholds, reversed):
    """
    Method that calculates the MATVs for the given threshold methods that are available in the scene
    """
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    self.matvRows = []

    extractor = self.createExtractor('matv')
    pixelVolume, pixelSpacing = self.getCubicCmPerPixel(reversed)
    suvImage = sitk.GetImageFromArray(self.suvMap)
    suvImage.SetSpacing(pixelSpacing)

    segmentationNodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    for thresh in thresholds:
      for segment in segmentationNodes:
        segmentName = segment.GetName()
        if thresh in segmentName:
          segmentNode = slicer.util.getNode(segmentName)
          segmentArray = self.getArrayFromSegmentationNode(self.petVolume, segmentNode)
          matv = self.getMATV(extractor, segmentArray, pixelVolume, suvImage, pixelSpacing)
          self.matvRows.append({
            'Segmentation': segmentName,
            'Voxel Volume': matv[0],
            'Mesh Volume': matv[1]
          })
    volumeDf = self.pd.DataFrame(self.matvRows)
    savePath = "/".join(self.petSeriesPath.split('/')[:-1])
    volumeFilePath = f'{savePath}/MATV_patient_{self.patientID}.xlsx'
    volumeDf.to_excel(volumeFilePath, index=False)
    qt.QApplication.setOverrideCursor(qt.Qt.ArrowCursor)
    slicer.util.infoDisplay(f'MATV calculations finished, MATVs stored at: {volumeFilePath}', 'MATVs extracted')

  def extractFeatures(self, thresholds, reversed):
    featuresRows = []
    extractor = self.createExtractor('all')
    pixelVolume, pixelSpacing = self.getCubicCmPerPixel(reversed)
    suvImage = sitk.GetImageFromArray(self.suvMap)
    suvImage.SetSpacing(pixelSpacing)
    peakSphere = self.createPeakSphere()
    segmentationNodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')

    for thresh in thresholds:
      for segment in segmentationNodes:
        segmentName = segment.GetName()
        if thresh in segmentName:
          singleFeaturesRows = []
          segmentNode = slicer.util.getNode(segmentName)

          # for converting
          segmentNode.CreateDefaultDisplayNodes()
          segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(self.petVolume)

          segmentations = segmentNode.GetSegmentation()
          segmentationNames = segmentations.GetSegmentIDs()
          # Individual segmentations
          for segName in segmentationNames:
            try:
              subSegmentArray = self.getArrayFromSegmentationNode(self.petVolume, segmentNode, segName)
              segName = segmentations.GetSegment(segName).GetName()
            except:
              # no segmentation result
              continue

            singleFeaturesRows = self.extractSingleSegmentFeatures(subSegmentArray, segName, suvImage,
                                                                   pixelSpacing, peakSphere, singleFeaturesRows,
                                                                   extractor)

          singleFeaturesDf = self.pd.DataFrame(singleFeaturesRows)
          savePath = "/".join(self.petSeriesPath.split('/')[:-1])
          singleFeaturesFile = f'{savePath}/PET_features_patient_{self.patientID}_{thresh}.xlsx'
          singleFeaturesDf.to_excel(singleFeaturesFile, index=False)

          # Total features
          try:
            segmentArray = self.getArrayFromSegmentationNode(self.petVolume, segmentNode)
          except:
            # no segmentation result
            continue

          segmentImage = sitk.GetImageFromArray(segmentArray)
          segmentImage.SetSpacing(pixelSpacing)

          featuresRow = {
            'Segmentation': segmentName
          }
          featureVector = extractor.execute(suvImage, segmentImage)
          for feature in featureVector.keys():
            if feature.find('original') == 0:
              value = float(featureVector[feature])
              featureDesc = feature[9:].replace("firstorder", "SUV").replace("shape_", "").replace("Voxel", "")
              if "Volume" in featureDesc:
                featureDesc += " (cc)"
                featuresRow[featureDesc] = value / 1000
              else:
                featuresRow[featureDesc] = value
            else:
              featuresRow[feature] = featureVector[feature]

          # Calculate SUVpeak
          segmentSuv = self.suvMap.copy()
          segmentSuv[segmentArray < 1.0] = 0.0
          suvPeak = self.calculateSuvPeak(peakSphere, segmentSuv)
          pos = list(featuresRow.keys()).index('SUV_Median') + 1
          items = list(featuresRow.items())
          items.insert(pos, ('SUV_Peak', suvPeak))
          featuresRow = dict(items)
          # TLG
          featuresRow["TLG"] = featuresRow["Volume (cc)"] * featuresRow["SUV_Mean"]

          # Dissemination features
          lesionsList = singleFeaturesDf.to_dict('records')
          nrOfLesions = len(lesionsList)
          featuresRow["Number of lesions"] = nrOfLesions
          dmaxPatient, spreadPatient = self.getDmaxAndSpreadPatient(lesionsList, nrOfLesions, pixelSpacing)
          featuresRow["Dmax Patient (mm)"] = dmaxPatient
          featuresRow["Spread Patient (mm)"] = spreadPatient
          max_index = singleFeaturesDf['Volume (cc)'].idxmax()
          largest_lesion = singleFeaturesDf.loc[max_index][2:]
          dmaxBulk, spreadBulk = self.getDmaxAndSpreadBulk(lesionsList, largest_lesion, pixelSpacing)
          featuresRow["Dmax Bulk (mm)"] = dmaxBulk
          featuresRow["Spread Bulk (mm)"] = spreadBulk

          featuresRows.append(featuresRow)

    featuresDf = self.pd.DataFrame(featuresRows)
    savePath = "/".join(self.petSeriesPath.split('/')[:-1])
    volumeFilePath = f'{savePath}/PET_features_patient_{self.patientID}_total.xlsx'
    featuresDf.to_excel(volumeFilePath, index=False)
    slicer.util.infoDisplay(f'PET feature extraction finished, features stored at: {volumeFilePath}',
                            'PET features extracted')

  def getDmaxAndSpreadPatient(self, lesions, lesionsNr, pixelSpacing):
    """
    Method that calculates the distance between the given
    lesions that are the furthest apart (D-max patient).
    Also calculates the largest value (over all lesions)
    of the sum of the distances from a lesion to all
    the other lesions (SPREAD patient).
    """
    max_distance = 0.0
    sums_of_distances = []
    for i in range(lesionsNr):
      distances_list = []
      location_a = lesions[i]

      if i < lesionsNr - 1:
        others = lesions[:i] + lesions[i + 1:]
      else:
        others = lesions[:i]

      for location_b in others:
        distance = self.calculate_distance(location_a['diagnostics_Mask-original_CenterOfMassIndex'],
                                           location_b['diagnostics_Mask-original_CenterOfMassIndex'], pixelSpacing)
        distances_list.append(distance)
        if distance > max_distance:
          max_distance = distance
      sums_of_distances.append(sum(distances_list))

    return max_distance, max(sums_of_distances)

  def getDmaxAndSpreadBulk(self, lesions, largest_lesion, pixelSpacing):
    """
    Method that calculates the distance between the given largest lesion and the lesion
    farthest away from that bulk (D-max bulk). Also calculates the sum of distances of that
    bulk from all other lesions (SPREAD bulk).
    """
    largest_lesion_location = largest_lesion['diagnostics_Mask-original_CenterOfMassIndex']
    distances = []
    for lesion in lesions:
      distance = self.calculate_distance(largest_lesion_location,
                                         lesion['diagnostics_Mask-original_CenterOfMassIndex'], pixelSpacing)
      distances.append(distance)
    return max(distances), sum(distances)

  def calculate_distance(self, location_a, location_b, pixel_spacing):
    """
    Method that calculates the Euclidean distance in mm between two given 3D points.
    """
    z_index_a, y_index_a, x_index_a = location_a
    z_index_b, y_index_b, x_index_b = location_b
    z_diff = z_index_a - z_index_b
    y_diff = y_index_a - y_index_b
    x_diff = x_index_a - x_index_b
    spacing_x, spacing_y, spacing_z = pixel_spacing

    distance = np.sqrt((x_diff * spacing_x) ** 2 +
                       (y_diff * spacing_y) ** 2 +
                       (z_diff * spacing_z) ** 2)

    return distance

  def extractSingleSegmentFeatures(self, subSegmentArray, segmentName, suvImage, pixelSpacing, peakSphere, singleFeaturesRows, extractor):
    segmentImage = sitk.GetImageFromArray(subSegmentArray)
    segmentImage.SetSpacing(pixelSpacing)

    featuresRow = {
      'Segmentation': segmentName
    }

    try:
      featureVector = extractor.execute(suvImage, segmentImage)
    except ValueError:
      # Nothing is segmented
      return singleFeaturesRows

    for feature in featureVector.keys():
      if feature.find('original') == 0:
        value = float(featureVector[feature])
        featureDesc = feature[9:].replace("firstorder", "SUV").replace("shape_", "").replace("Voxel", "")
        if "Volume" in featureDesc:
          featureDesc += " (cc)"
          featuresRow[featureDesc] = value / 1000
        else:
          featuresRow[featureDesc] = value
      else:
        featuresRow[feature] = featureVector[feature]

    segmentSuv = self.suvMap.copy()
    segmentSuv[subSegmentArray < 1.0] = 0.0
    # Calculate SUVpeak
    suvPeak = self.calculateSuvPeak(peakSphere, segmentSuv)
    pos = list(featuresRow.keys()).index('SUV_Median') + 1

    items = list(featuresRow.items())
    items.insert(pos, ('SUV_Peak', suvPeak))
    featuresRow = dict(items)

    # TLG
    featuresRow["TLG"] = featuresRow["Volume (cc)"] * featuresRow["SUV_Mean"]

    singleFeaturesRows.append(featuresRow)
    return singleFeaturesRows

  def extractVOIsMetrics(self, reversed):
    pixelVolume, pixelSpacing = self.getCubicCmPerPixel(reversed)
    savePath = "/".join(self.petSeriesPath.split('/')[:-1])
    fileName = f"VOIs_metrics_patient_{self.patientID}.xlsx"
    filePath = f'{savePath}/{fileName}'

    if os.path.exists(filePath):
      if slicer.util.confirmOkCancelDisplay(f"File '{fileName}' already exists, do you want to overwrite it?"):
        self.performVOImetricsExtraction(filePath, pixelSpacing)
      else:
        slicer.util.infoDisplay(f'VOIs metrics calculation canceled.',
                                'Canceled')
    else:
      self.performVOImetricsExtraction(filePath, pixelSpacing)

  def performVOImetricsExtraction(self, filePath, pixelSpacing):
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    featuresRows = []
    extractor = self.createExtractor('VOIs')
    suvImage = sitk.GetImageFromArray(self.suvMap)
    suvImage.SetSpacing(pixelSpacing)

    peakSphere = self.createPeakSphere()
    vois = slicer.mrmlScene.GetNodesByClass('vtkMRMLSegmentationNode')
    for voi in vois:
      voiName = voi.GetName()
      if voiName.startswith('VOI_'):
        voiNode = slicer.util.getNode(voiName)
        voiArray = self.getArrayFromSegmentationNode(self.petVolume, voiNode)
        voiImage = sitk.GetImageFromArray(voiArray)
        voiImage.SetSpacing(pixelSpacing)
        featuresRow = {
          'VOI': voiName
        }
        featureVector = extractor.execute(suvImage, voiImage)
        for feature in featureVector.keys():
          if feature.find('original') == 0:
            value = float(featureVector[feature])
            featureDesc = feature[9:].replace("firstorder", "SUV").replace("shape_", "").replace("Voxel", "")
            if "Volume" in featureDesc:
              featureDesc += " (cc)"
              featuresRow[featureDesc] = value / 1000
            else:
              featuresRow[featureDesc] = value

        # Calculate SUVpeak
        voiSuv = self.suvMap.copy()
        voiSuv[voiArray < 1.0] = 0.0
        suvPeak = self.calculateSuvPeak(peakSphere, voiSuv)
        pos = list(featuresRow.keys()).index('SUV_Median') + 1
        items = list(featuresRow.items())
        items.insert(pos, ('SUV_Peak', suvPeak))
        featuresRow = dict(items)
        # TLG
        featuresRow["TLG"] = featuresRow["Volume (cc)"] * featuresRow["SUV_Mean"]
        featuresRows.append(featuresRow)

    featuresDf = self.pd.DataFrame(featuresRows)
    self.saveMetrics(featuresDf, filePath)

  def saveMetrics(self, df, filePath):
    qt.QApplication.setOverrideCursor(qt.Qt.ArrowCursor)
    try:
      df.to_excel(filePath, index=False)
      slicer.util.infoDisplay(f'Metrics calculated, stored at: {filePath}',
                              'Metrics extracted')
    except PermissionError:
      slicer.util.infoDisplay(f'Could not save metrics, file "{filePath}" is opened.',
                              'Permission denied')

  def createPeakSphere(self):
    pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    pointListNode.SetName("peakSphere")

    bounds = [0, 0, 0, 0, 0, 0]
    self.petVolume.GetRASBounds(bounds)
    nodeCenter = [round((bounds[0] + bounds[1]) / 2), round((bounds[2] + bounds[3]) / 2), round((bounds[4] + bounds[5]) / 2)]
    pointListNode.AddControlPoint(nodeCenter)

    self.createSphere(1.2, "peakSphere", False)
    peakNode = slicer.util.getNode("peakSphere_")
    peakArray = self.getArrayFromSegmentationNode(self.petVolume, peakNode)
    try:
      slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(peakArray))
      trimmedPeak = peakArray[slices]
    except:
      # estimate SUVpeak
      center = (1, 1, 1)
      size = (4, 4, 4)
      distance = np.linalg.norm(np.subtract(np.indices(size).T, np.asarray(center)), axis=len(center))
      trimmedPeak = np.ones(size) * (distance <= 1)

    slicer.mrmlScene.RemoveNode(pointListNode)
    slicer.mrmlScene.RemoveNode(peakNode)

    return trimmedPeak

  def calculateSuvPeak(self, sphere, suvArray):
    sphereShape = sphere.shape
    sphereVolume = np.sum(sphere > 0)

    # `suv_array` and `sphere` should be dimension (1, 1, z, y, x) for Conv3d layer
    suvArrayDl = np.expand_dims(suvArray, 0)
    suvArrayDl = np.expand_dims(suvArrayDl, 0)
    sphereDl = np.expand_dims(sphere, 0)
    sphereDl = np.expand_dims(sphereDl, 0)

    # Map to tensor
    suvArrayDl = self.torchTensor(suvArrayDl)
    sphereDl = self.torchTensor(sphereDl)

    model = self.Conv3d(in_channels=1, out_channels=1, kernel_size=sphereShape, stride=1, padding=0)

    # Replace random Conv3d filter by sphere weights
    modelDict = model.state_dict()
    modelDict['weight'] = sphereDl
    modelDict['bias'] = self.torchTensor([0])
    model.load_state_dict(modelDict)

    out = model(suvArrayDl.float())
    out = out[0, 0].max() / sphereVolume

    return out.item()

  def createExtractor(self, method):
    extractor = self.featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    if method == 'all':
      shapeFeatures = ['MeshVolume', 'VoxelVolume', 'Compactness1', 'Compactness2', 'Elongation', 'Flatness',
                       'LeastAxisLength', 'MajorAxisLength', 'Maximum2DDiameterColumn', 'Maximum2DDiameterRow',
                       'Maximum2DDiameterSlice', 'Maximum3DDiameter', 'MinorAxisLength', 'SphericalDisproportion',
                       'Sphericity', 'SurfaceArea', 'SurfaceVolumeRatio']
      firstorderFeatures = ['10Percentile', '90Percentile', 'Energy', 'Entropy', 'InterquartileRange', 'Kurtosis',
                            'Maximum', 'MeanAbsoluteDeviation', 'Mean', 'Median', 'Minimum', 'Range',
                            'RobustMeanAbsoluteDeviation',
                            'RootMeanSquared', 'Skewness', 'StandardDeviation', 'TotalEnergy', 'Uniformity', 'Variance']
      extractor.enableFeaturesByName(shape=shapeFeatures, firstorder=firstorderFeatures)
    elif method == 'matv':
      extractor.enableFeaturesByName(shape=['MeshVolume'])
    elif method == 'VOIs':
      shapeFeatures = ['VoxelVolume', 'Sphericity']
      firstorderFeatures = ['Maximum', 'Minimum', 'Mean', 'Median', 'Kurtosis', 'Skewness']
      extractor.enableFeaturesByName(firstorder=firstorderFeatures, shape=shapeFeatures)
    return extractor

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
    for i in range(0, len(seeds)):
      seed = seeds[i]
      if thresholdDescr == 'A50P':
        self.suvThreshold = self.getPeakBackgroundThreshold(i, suvThreshold)
      # perform the segmentation for the given seed
      segmentImage, segmentArray = self.seedGrowSegmentation(seed)
      # save the segmentation binary array
      if np.count_nonzero(segmentArray) > 1:
        # set created segment to zero for next seed
        self.suvImageArray[segmentArray == 1] = 0.0
        # convert segmentation image to a labelmap node
        labelmapNode = self.createLabelMapNode(origin, segmentImage, spacing)
        labelmapNodes.append(labelmapNode)
    if len(labelmapNodes) > 0:
      self.convertNodesToSegmentationNode(labelmapNodes, True, False, 'False', thresholdDescr)

  def getPeakBackgroundThreshold(self, i, bgValue):
    peakSphere = self.createPeakSphere()
    voiNode = slicer.util.getNode(f'{i}_A50P_')
    voiArray = self.getArrayFromSegmentationNode(self.petVolume, voiNode)

    # Calculate SUVpeak
    voiSuv = self.suvMap.copy()
    voiSuv[voiArray < 1.0] = 0.0
    suvPeak = self.calculateSuvPeak(peakSphere, voiSuv)
    slicer.mrmlScene.RemoveNode(voiNode)

    # Calculate A50P threshold
    threshold = 0.5 * (suvPeak + bgValue)

    return threshold

  def createSuvMaxSegmentation(self, suvMap, suvPerRoi, shape, origin, spacing, petVolume, roiFilter):
    """
    Method that creates the segmentation result based on SUVmax thresholding
    """
    roisCoords = self.getRoisCoords(petVolume)

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
      self.convertNodesToSegmentationNode(labelmapNodes, True, False, 'False', '41suvMax')

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
        self.convertNodesToSegmentationNode([labelMapNode], True, False, 'False', '41suvMax')

  def createSuvMaxSeedSegmentation(self, suvMap, seeds, shape, origin, spacing):
    """
    Method that creates the segmentation result based on SUVmax thresholding
    """
    labelmapNodes = []
    self.maxX, self.maxY, self.maxZ = shape[2], shape[1], shape[0]
    self.suvImageArray = suvMap.copy()
    self.thresholdDescr = '41suvMaxSeed'
    for i in range(0, len(seeds)):
      seed = seeds[i]

      # Calculate 41suvMaxSeed threshold
      voiNode = slicer.util.getNode(f'{i}_41suvMaxSeed_')
      voiArray = self.getArrayFromSegmentationNode(self.petVolume, voiNode)
      voiSuv = self.suvMap.copy()
      voiSuv[voiArray < 1.0] = 0.0
      slicer.mrmlScene.RemoveNode(voiNode)
      threshold = 0.41 * np.max(voiSuv)
      self.suvThreshold = threshold

      # perform the segmentation
      segmentImage, segmentArray = self.seedGrowSegmentation(seed)
      if np.count_nonzero(segmentArray) > 1:
        self.suvImageArray[segmentArray == 1] = 0.0
        labelmapNode = self.createLabelMapNode(origin, segmentImage, spacing)
        labelmapNodes.append(labelmapNode)
    if len(labelmapNodes) > 0:
      self.convertNodesToSegmentationNode(labelmapNodes, True, False, 'False', '41suvMaxSeed')

  def createMajVotingSegmentation(self, segmentationsForMajVoting, mvMethods, origin, spacing, segmentationMethods,
                                  petVolume):
    """
    Method that creates the segmentation result based on majority voting
    """
    for method in segmentationMethods:
      try:
        segmentNode = slicer.util.getNode('{0}_segmentation_{1}'.format(self.patientID, method))
      except:
        return False
      try:
        segmentArray = self.getArrayFromSegmentationNode(petVolume, segmentNode)
        segmentationsForMajVoting.append(segmentArray)
      except AttributeError:
        # method created no segmentation
        continue

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
          self.convertNodesToSegmentationNode([labelMapNode], True, False, 'False', method)
      return True

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

  def getCubicCmPerPixel(self, reversed):
    """
    Method that computes the cubic cm per pixel for the given PET DICOM series
    """
    self.setPetDataParameters(reversed)
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

  def setPetDataParameters(self, reversed):
    self.petVolume = self.getVolumeFromList('pet')
    petImageFileList = self.getPetFilesLists(self.petVolume, reversed)
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
    liverSuvValues = []
    liverConditions = ['liverSUVmax' in segmentationMethods, 'PERCIST' in segmentationMethods]
    if any(liverConditions):
      liverSuvValues = self.getSphereSuvValues(petVolume, suvImageArray, "VOI_liver")

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
        liverAvr = np.mean(liverSuvValues)
        liverStd = np.std(liverSuvValues)
        percistSuv = 1.5 * liverAvr + (2 * liverStd)
        thresholds[method] = percistSuv
      elif method == 'A50P':
        lungSuvValues = self.getSphereSuvValues(petVolume, suvImageArray, "VOI_lung")
        if len(liverSuvValues) == 0:
          liverSuvValues = self.getSphereSuvValues(petVolume, suvImageArray, "VOI_liver")
          liverAvr = np.mean(liverSuvValues)
        bgCorrection = (np.mean(lungSuvValues) + liverAvr) / 2
        thresholds[method] = bgCorrection
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

  def getArrayFromSegmentationNode(self, volumeNode, segmentsNode, fromSubSegment=False):
    """
    Method that gets all the segmentations from the given node name
    """
    if fromSubSegment != False:
      segmentationsArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentsNode, fromSubSegment, volumeNode)
    else:
      segmentsNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
      labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
      slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentsNode, labelmapVolumeNode,
                                                                           slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
      segmentationsArray = slicer.util.arrayFromVolume(labelmapVolumeNode)
      slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

    # set segment pixels to 1.0
    segmentationsArray[segmentationsArray > 0] = 1.0

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
    # get patient and scan info
    self.patientID = dicomSeries.PatientID
    try:
      self.patientAge = int(dicomSeries.PatientAge[:-1])
    except AttributeError:
      self.patientAge = int((datetime.datetime.strptime(dicomSeries.StudyDate, '%Y%m%d') -
                             datetime.datetime.strptime(dicomSeries.PatientBirthDate, '%Y%m%d')).days)
    self.PixelSpacing = dicomSeries.PixelSpacing
    self.SliceThickness = dicomSeries.SliceThickness

    try:
      # get patient weight (grams)
      weight = float(dicomSeries.PatientWeight) * 1000
      # start time for the Radiopharmaceutical Injection
      rpStartTime = dicomSeries.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
      if '.' not in rpStartTime:
        rpStartTime += ".0"
      injectionTime = datetime.datetime.strptime(rpStartTime, '%H%M%S.%f')
      # half life for Radionuclide (seconds)
      halfLife = float(dicomSeries.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
      # total dose injected for Radionuclide (Becquerels Bq)
      injectedDose = float(dicomSeries.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

      # get scan time
      acqTime = dicomSeries.AcquisitionTime
      if '.' not in acqTime:
        acqTime += ".0"
      scantime = datetime.datetime.strptime(acqTime, '%H%M%S.%f')
      # calculate decay
      decay = np.exp(-np.log(2) * ((scantime - injectionTime).seconds) / halfLife)
      # calculate the dose decayed during procedure (Bq)
      injectedDoseDecay = injectedDose * decay
    except:
      # make estimation
      traceback.print_exc()
      weight = 75000
      isEstimated = True
      decay = np.exp(-np.log(2) * (1.75 * 3600) / 6588)  # 90 min waiting time, 15 min preparation
      injectedDoseDecay = 420000000 * decay  # 420 MBq

    suvMap = imageArray * weight / injectedDoseDecay

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

  def getPaintedSeedPoints(self, refVolume, segmentationMethods):
    """
    Method that retrieves the user provided painted seeds
    """
    seedCoordinates = []
    for i in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsFiducialNode")):
      pointList = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLMarkupsFiducialNode")
      numControlPoints = pointList.GetNumberOfControlPoints()
      for i in range(numControlPoints):
        isVisible = pointList.GetNthControlPointVisibility(i)
        if isVisible:
          zxyCoords = self.getSeedCoordinate(pointList, i, refVolume)
          seedCoordinates.append(zxyCoords)
          if 'A50P' in segmentationMethods:
            rasCoords = pointList.GetNthControlPointPosition(i)
            self.createSphere(3, f'{i}_A50P', rasCoords)
          if '41suvMaxSeed' in segmentationMethods:
            rasCoords = pointList.GetNthControlPointPosition(i)
            self.createSphere(1.2, f'{i}_41suvMaxSeed', rasCoords)

    return seedCoordinates

  def getSeedCoordinate(self, pointList, i, refVolume):
    """
    Method that retrieves the IJK coordinate from a given control point index in the provided point list
    and reference volume.
    """
    # Get RAS coordinates of seed
    rasCoords = pointList.GetNthControlPointPosition(i)
    # Convert RAS to ZXY
    zxyCoords = self.convertRasPointToIjkPoint(rasCoords, refVolume)

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

  def getRoisCoords(self, petVolume):
    """
    Method that gets the coordinates of all visible ROIs in the scene
    """
    roisCoords = {}
    for i in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsROINode")):
      roi = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLMarkupsROINode")
      if roi.GetDisplayVisibility():
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
    # Get RAS coordinates
    center = [0, 0, 0]
    radius = [0, 0, 0]
    roi.GetXYZ(center)
    roi.GetRadiusXYZ(radius)
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
    self.tempDataDir = os.path.join(slicer.app.temporaryPath, 'MUSTSegmenterTest')
    self.tempDicomDatabaseDir = os.path.join(slicer.app.temporaryPath, 'MUSTSegmenterTestDicom')

    # segmentation parameters
    self.segmentationLogic = MUSTsegmenterLogic()
    self.organSegments = None
    self.suvPerRoi = False
    self.roiFilter = False
    self.reversed = False
    self.segmentationColors = {
      'suv2.5': [0.12, 0.55, 0.18],
      'suv3.0': [0.0, 0.8, 0.2],
      'suv4.0': [0.55, 0.82, 0.35],
      '41suvMax': [0.22, 0.08, 0.94],
      '41suvMaxSeed': [0.22, 0.08, 0.94],
      'A50P': [0.26, 0.16, 0.79],
      'liverSUVmax': [0.08, 0.37, 0.94],
      'PERCIST': [0.04, 0.60, 0.87],
      'MV2': [0.65, 0.0, 0.0],
      'MV3': [1.0, 0.48, 0.48]
    }
    self.segmentationMethods = ['suv2.5', 'suv3.0', 'suv4.0',
                                '41suvMax', '41suvMaxSeed', 'liverSUVmax', 'PERCIST',
                                'A50P', 'MV2', 'MV3']

  def runTest(self):
    self.setUp()
    self.testSegmentation()

  def testSegmentation(self):
    self.delayDisplay('Starting the test')
    self.loadTestData()
    self.performSegmentationTests()

  def performSegmentationTests(self):
    self.segmentationLogic.performSegmentation(self.organSegments, self.segmentationMethods, self.suvPerRoi,
                                               self.roiFilter, self.reversed, self.segmentationColors)
    self.segmentationLogic.extractVOIsMetrics(self.reversed)
    self.segmentationLogic.calulateMATV(self.segmentationMethods, False)
    self.segmentationLogic.extractFeatures(self.segmentationMethods, False)

  def loadTestData(self):
    zipUrl = "https://github.com/kyliekeijzer/Slicer-PET-MUST-segmenter/raw/master/Sample%20Data/Sample%20Data.zip"
    zipFilePath = os.path.join(self.tempDataDir, 'dicom.zip')
    zipFileData = os.path.join(self.tempDataDir, 'dicom')

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
    petPath = os.path.join(zipFileData, 'Sample Data', 'PET')
    DICOMUtils.importDicom(petPath)

    # Load PET
    dicomFiles = slicer.util.getFilesInDirectory(petPath)
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
    sampleDataPath = os.path.join(zipFileData, 'Sample Data')
    slicer.util.loadMarkups(os.path.join(sampleDataPath, 'seed.mrk.json'))
    import json
    roiNames = ['R', 'R_1', 'R_2', 'R_3']
    for r in roiNames:
      rObj = open(os.path.join(sampleDataPath, r + '.json'))
      coords = json.load(rObj)
      roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
      roi.SetName(r)
      roi.SetRadiusXYZ(list(coords['radius']))
      center = list(coords['center'])
      roi.SetXYZ(center[0], center[1], center[2])
    # load liverSphere
    liverSphere = slicer.util.loadSegmentation(os.path.join(sampleDataPath, 'VOI_liver.seg.vtm'))
    liverSphere.SetName('VOI_liver')
    lungSphere = slicer.util.loadSegmentation(os.path.join(sampleDataPath, 'VOI_lung.seg.vtm'))
    lungSphere.SetName('VOI_lung')

