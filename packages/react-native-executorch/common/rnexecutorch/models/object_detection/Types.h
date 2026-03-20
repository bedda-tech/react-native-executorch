#pragma once

#include <cstdint>
#include <rnexecutorch/data_processing/CVTypes.h>
#include <string>

namespace rnexecutorch::models::object_detection::types {
struct Detection {

  Detection() = default;
  Detection(cv_processing::BBox bbox, std::string label, int32_t classIndex,
            float score)
      : bbox(bbox), label(std::move(label)), classIndex(classIndex),
        score(score) {}

  cv_processing::BBox bbox;
  std::string label;
  int32_t classIndex;
  float score;
};

} // namespace rnexecutorch::models::object_detection::types
