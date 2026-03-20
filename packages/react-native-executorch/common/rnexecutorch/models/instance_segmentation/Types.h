#pragma once

#include <cstdint>
#include <memory>
#include <rnexecutorch/data_processing/CVTypes.h>
#include <rnexecutorch/jsi/OwningArrayBuffer.h>

namespace rnexecutorch::models::instance_segmentation::types {

/**
 * Represents a single detected instance in instance segmentation output.
 *
 * Contains bounding box coordinates, binary segmentation mask, class label,
 * and confidence score.
 */
struct Instance {

  Instance() = default;
  Instance(cv_processing::BBox bbox, std::shared_ptr<OwningArrayBuffer> mask,
           int32_t maskWidth, int32_t maskHeight, int32_t classIndex,
           float score)
      : bbox(bbox), mask(std::move(mask)), maskWidth(maskWidth),
        maskHeight(maskHeight), classIndex(classIndex), score(score) {}

  cv_processing::BBox bbox;
  std::shared_ptr<OwningArrayBuffer> mask;
  int32_t maskWidth;
  int32_t maskHeight;
  int32_t classIndex;
  float score;
};

} // namespace rnexecutorch::models::instance_segmentation::types
