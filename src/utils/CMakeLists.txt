project(rs_utils)

# TODO: get rid of it
include_directories(BEFORE ${PCL_INCLUDE_DIRS})
add_library(rs_utils SHARED
  src/BlurDetector.cpp
  src/DepthImageProcessing.cpp
  src/GeometryCV.cpp
  src/Drawing.cpp
)
target_link_libraries(rs_utils rs_core)

rs_add_library(rs_BlurDetectorAnnotator src/BlurDetectorAnnotator.cpp)
target_link_libraries(rs_BlurDetectorAnnotator rs_core rs_utils)

rs_add_library(rs_PointCloudDownsampler src/PointCloudDownsampler.cpp)
target_link_libraries(rs_PointCloudDownsampler rs_core)

rs_add_library(rs_ClusterMerger src/ClusterMerger.cpp)
target_link_libraries(rs_ClusterMerger rs_core)

rs_add_library(rs_ImagePreprocessor src/ImagePreprocessor.cpp)
target_link_libraries(rs_ImagePreprocessor rs_core rs_utils rs_segmentation)

rs_add_library(rs_Trigger src/Trigger.cpp)
target_link_libraries(rs_Trigger rs_core rs_io)

add_library(rs_analysisEngineManager
    src/RSAnalysisEngine.cpp 
    src/RSPipelineManager.cpp)
target_link_libraries(rs_analysisEngineManager rs_core rs_io ${LIBAPR_LIBRARY} ${UIMA_LIBRARY} ${ICUUC_LIBRARY} ${catkin_LIBRARIES})

rs_add_executable(run src/run.cpp)
target_link_libraries(run rs_analysisEngineManager)

