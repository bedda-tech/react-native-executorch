#include "CVProcessing.h"
#include <algorithm>
#include <cmath>
#include <rnexecutorch/Error.h>
#include <rnexecutorch/ErrorCodes.h>
#include <rnexecutorch/Log.h>

namespace rnexecutorch::cv_processing {

float computeIoU(const BBox &a, const BBox &b) {
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);

  float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float areaA = a.area();
  float areaB = b.area();
  float unionArea = areaA + areaB - intersectionArea;

  return (unionArea > 0.0f) ? (intersectionArea / unionArea) : 0.0f;
}

std::optional<cv::Scalar> validateNormParam(const std::vector<float> &values,
                                            const char *paramName) {
  if (values.size() == 3) {
    return cv::Scalar(values[0], values[1], values[2]);
  } else if (!values.empty()) {
    log(LOG_LEVEL::Warn,
        std::string(paramName) +
            " must have 3 elements — ignoring provided value.");
  }
  return std::nullopt;
}

std::set<int32_t>
prepareAllowedClasses(const std::vector<int32_t> &classIndices) {
  std::set<int32_t> allowedClasses;
  if (!classIndices.empty()) {
    allowedClasses.insert(classIndices.begin(), classIndices.end());
  }
  return allowedClasses;
}

void validateThresholds(double confidenceThreshold, double iouThreshold) {
  if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
    throw RnExecutorchError(RnExecutorchErrorCode::InvalidConfig,
                            "Confidence threshold must be in range [0, 1].");
  }

  if (iouThreshold < 0.0 || iouThreshold > 1.0) {
    throw RnExecutorchError(RnExecutorchErrorCode::InvalidConfig,
                            "IoU threshold must be in range [0, 1].");
  }
}

} // namespace rnexecutorch::cv_processing
