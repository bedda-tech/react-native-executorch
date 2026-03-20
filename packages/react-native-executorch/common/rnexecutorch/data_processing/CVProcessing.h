#pragma once

#include "CVTypes.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <optional>
#include <set>
#include <vector>

namespace rnexecutorch::cv_processing {

/**
 * @brief Compute Intersection over Union (IoU) between two bounding boxes
 * @param a First bounding box
 * @param b Second bounding box
 * @return IoU value between 0.0 and 1.0
 *
 * Moved from utils/computer_vision/Processing.h for consolidation.
 */
float computeIoU(const BBox &a, const BBox &b);

/**
 * @brief Non-Maximum Suppression for detection/segmentation results
 * @tparam T Type that has bbox and score fields (satisfies HasBBoxAndScore)
 * @param items Vector of items to filter
 * @param iouThreshold IoU threshold for suppression (typically 0.5)
 * @return Filtered vector with overlapping detections removed
 *
 * Moved from utils/computer_vision/Processing.h for consolidation.
 * Handles both class-aware and class-agnostic NMS automatically.
 */
template <HasBBoxAndScore T>
std::vector<T> nonMaxSuppression(std::vector<T> items, double iouThreshold) {
  if (items.empty()) {
    return {};
  }

  // Sort by score in descending order
  std::ranges::sort(items,
                    [](const T &a, const T &b) { return a.score > b.score; });

  std::vector<T> result;
  std::vector<bool> suppressed(items.size(), false);

  for (size_t i = 0; i < items.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }

    result.push_back(items[i]);

    // Suppress overlapping boxes
    for (size_t j = i + 1; j < items.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      // If type has classIndex, only suppress boxes of same class
      if constexpr (requires(T t) { t.classIndex; }) {
        if (items[i].classIndex != items[j].classIndex) {
          continue;
        }
      }

      float iou = computeIoU(items[i].bbox, items[j].bbox);
      if (iou > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

/**
 * @brief Validate and convert normalization parameter vector to cv::Scalar
 * @param values Vector of normalization values (should have 3 elements for RGB)
 * @param paramName Parameter name for logging (e.g., "normMean", "normStd")
 * @return Optional cv::Scalar if valid (3 elements), nullopt otherwise
 *
 * Replaces duplicate validation logic across ObjectDetection,
 * BaseInstanceSegmentation, and BaseSemanticSegmentation.
 */
std::optional<cv::Scalar> validateNormParam(const std::vector<float> &values,
                                            const char *paramName);

/**
 * @brief Convert class indices vector to a set for efficient filtering
 * @param classIndices Vector of class indices to allow
 * @return Set of allowed class indices (empty set = allow all classes)
 *
 * Used by detection and segmentation models to filter results by class.
 */
std::set<int32_t>
prepareAllowedClasses(const std::vector<int32_t> &classIndices);

/**
 * @brief Validate confidence and IoU thresholds are in valid range [0, 1]
 * @param confidenceThreshold Detection confidence threshold
 * @param iouThreshold Non-maximum suppression IoU threshold
 * @throws RnExecutorchError if either threshold is out of range
 *
 * Used by detection and segmentation models to validate user input.
 */
void validateThresholds(double confidenceThreshold, double iouThreshold);

} // namespace rnexecutorch::cv_processing
