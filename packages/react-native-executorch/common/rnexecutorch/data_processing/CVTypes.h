#pragma once

#include <concepts>
#include <opencv2/opencv.hpp>

namespace rnexecutorch::cv_processing {

/**
 * @brief Bounding box representation with x1, y1, x2, y2 coordinates
 *
 * Moved from utils/computer_vision/Types.h for consolidation.
 */
struct BBox {
  float x1, y1, x2, y2;

  float width() const { return x2 - x1; }

  float height() const { return y2 - y1; }

  float area() const { return width() * height(); }

  bool isValid() const {
    return x2 > x1 && y2 > y1 && x1 >= 0.0f && y1 >= 0.0f;
  }

  BBox scale(float widthRatio, float heightRatio) const {
    return {x1 * widthRatio, y1 * heightRatio, x2 * widthRatio,
            y2 * heightRatio};
  }
};

/**
 * @brief Concept for types that have a bounding box and confidence score
 *
 * Used for NMS and other detection/segmentation operations.
 */
template <typename T>
concept HasBBoxAndScore = requires(T t) {
  { t.bbox } -> std::convertible_to<BBox>;
  { t.score } -> std::convertible_to<float>;
};

/**
 * @brief Scale ratios for mapping between original and model input dimensions
 *
 * Replaces duplicate scale ratio calculation code across multiple models.
 */
struct ScaleRatios {
  float widthRatio;
  float heightRatio;

  /**
   * @brief Compute scale ratios from original size to model input size
   * @param original Original image dimensions
   * @param model Model input dimensions
   * @return ScaleRatios struct containing width and height ratios
   */
  static ScaleRatios compute(cv::Size original, cv::Size model) {
    return {static_cast<float>(original.width) / model.width,
            static_cast<float>(original.height) / model.height};
  }
};

} // namespace rnexecutorch::cv_processing
