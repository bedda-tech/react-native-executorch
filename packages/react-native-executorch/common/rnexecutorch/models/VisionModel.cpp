#include "VisionModel.h"
#include <rnexecutorch/Error.h>
#include <rnexecutorch/ErrorCodes.h>
#include <rnexecutorch/data_processing/CVProcessing.h>
#include <rnexecutorch/utils/FrameProcessor.h>
#include <rnexecutorch/utils/FrameTransform.h>

namespace rnexecutorch::models {

using namespace facebook;

VisionModel::VisionModel(const std::string &modelSource,
                         std::shared_ptr<react::CallInvoker> callInvoker)
    : BaseModel(modelSource, callInvoker) {}

void VisionModel::unload() noexcept {
  std::scoped_lock lock(inference_mutex_);
  BaseModel::unload();
}

cv::Size VisionModel::modelInputSize() const {
  // For multi-method models, query the currently loaded method's input shape
  if (!currentlyLoadedMethod_.empty()) {
    auto inputShapes = getAllInputShapes(currentlyLoadedMethod_);
    if (!inputShapes.empty() && !inputShapes[0].empty() &&
        inputShapes[0].size() >= 2) {
      const auto &shape = inputShapes[0];
      return {static_cast<int>(shape[shape.size() - 2]),
              static_cast<int>(shape[shape.size() - 1])};
    }
  }

  // Default: use cached modelInputShape_ from single-method models
  if (modelInputShape_.size() < 2) {
    return {0, 0};
  }
  return cv::Size(modelInputShape_[modelInputShape_.size() - 1],
                  modelInputShape_[modelInputShape_.size() - 2]);
}

cv::Mat VisionModel::extractFromFrame(jsi::Runtime &runtime,
                                      const jsi::Value &frameData) const {
  cv::Mat frame = ::rnexecutorch::utils::frameToMat(runtime, frameData);
  cv::Mat rgb;
#ifdef __APPLE__
  cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
#else
  cv::cvtColor(frame, rgb, cv::COLOR_RGBA2RGB);
#endif
  return rgb;
}

cv::Mat VisionModel::preprocess(const cv::Mat &image) const {
  const cv::Size targetSize = modelInputSize();
  if (image.size() == targetSize) {
    return image;
  }
  cv::Mat resized;
  cv::resize(image, resized, targetSize);
  return resized;
}

cv::Mat VisionModel::extractFromPixels(const JSTensorViewIn &tensorView) const {
  return ::rnexecutorch::utils::pixelsToMat(tensorView);
}

void VisionModel::ensureMethodLoaded(const std::string &methodName) {
  if (methodName.empty()) {
    throw RnExecutorchError(
        RnExecutorchErrorCode::InvalidConfig,
        "Method name cannot be empty. Use 'forward' for single-method models "
        "or 'forward_{inputSize}' for multi-method models.");
  }

  if (currentlyLoadedMethod_ == methodName) {
    return;
  }

  if (!module_) {
    throw RnExecutorchError(RnExecutorchErrorCode::ModuleNotLoaded,
                            "Model not loaded. Cannot load method '" +
                                methodName + "'.");
  }

  if (!currentlyLoadedMethod_.empty()) {
    module_->unload_method(currentlyLoadedMethod_);
  }

  auto loadResult = module_->load_method(methodName);
  if (loadResult != executorch::runtime::Error::Ok) {
    throw RnExecutorchError(
        loadResult, "Failed to load method '" + methodName +
                        "'. Ensure the method exists in the exported model.");
  }

  currentlyLoadedMethod_ = methodName;
}

void VisionModel::initializeNormalization(const std::vector<float> &normMean,
                                          const std::vector<float> &normStd) {
  normMean_ = cv_processing::validateNormParam(normMean, "normMean");
  normStd_ = cv_processing::validateNormParam(normStd, "normStd");
}

} // namespace rnexecutorch::models
