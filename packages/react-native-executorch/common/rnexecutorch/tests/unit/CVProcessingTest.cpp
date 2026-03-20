#include <gtest/gtest.h>
#include <rnexecutorch/Error.h>
#include <rnexecutorch/data_processing/CVProcessing.h>
#include <rnexecutorch/data_processing/CVTypes.h>

using namespace rnexecutorch::cv_processing;

class CVProcessingTest : public ::testing::Test {};

// ============================================================================
// prepareAllowedClasses Tests
// ============================================================================

TEST_F(CVProcessingTest, PrepareAllowedClasses_EmptyVector_ReturnsEmptySet) {
  std::vector<int32_t> input = {};
  auto result = prepareAllowedClasses(input);
  EXPECT_TRUE(result.empty());
}

TEST_F(CVProcessingTest, PrepareAllowedClasses_SingleClass_ReturnsSetWithOne) {
  std::vector<int32_t> input = {5};
  auto result = prepareAllowedClasses(input);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result.count(5) > 0);
}

TEST_F(CVProcessingTest,
       PrepareAllowedClasses_MultipleClasses_ReturnsCorrectSet) {
  std::vector<int32_t> input = {1, 3, 5, 7};
  auto result = prepareAllowedClasses(input);
  EXPECT_EQ(result.size(), 4);
  EXPECT_TRUE(result.count(1) > 0);
  EXPECT_TRUE(result.count(3) > 0);
  EXPECT_TRUE(result.count(5) > 0);
  EXPECT_TRUE(result.count(7) > 0);
}

TEST_F(CVProcessingTest,
       PrepareAllowedClasses_DuplicateClasses_RemovesDuplicates) {
  std::vector<int32_t> input = {1, 3, 3, 5, 1};
  auto result = prepareAllowedClasses(input);
  EXPECT_EQ(result.size(), 3); // Should have 1, 3, 5
  EXPECT_TRUE(result.count(1) > 0);
  EXPECT_TRUE(result.count(3) > 0);
  EXPECT_TRUE(result.count(5) > 0);
}

// ============================================================================
// validateThresholds Tests
// ============================================================================

TEST_F(CVProcessingTest, ValidateThresholds_ValidValues_DoesNotThrow) {
  EXPECT_NO_THROW(validateThresholds(0.5, 0.5));
  EXPECT_NO_THROW(validateThresholds(0.0, 0.0));
  EXPECT_NO_THROW(validateThresholds(1.0, 1.0));
}

TEST_F(CVProcessingTest, ValidateThresholds_NegativeConfidence_Throws) {
  EXPECT_THROW(validateThresholds(-0.1, 0.5), rnexecutorch::RnExecutorchError);
}

TEST_F(CVProcessingTest, ValidateThresholds_ConfidenceAboveOne_Throws) {
  EXPECT_THROW(validateThresholds(1.1, 0.5), rnexecutorch::RnExecutorchError);
}

TEST_F(CVProcessingTest, ValidateThresholds_NegativeIoU_Throws) {
  EXPECT_THROW(validateThresholds(0.5, -0.1), rnexecutorch::RnExecutorchError);
}

TEST_F(CVProcessingTest, ValidateThresholds_IoUAboveOne_Throws) {
  EXPECT_THROW(validateThresholds(0.5, 1.1), rnexecutorch::RnExecutorchError);
}

// ============================================================================
// computeIoU Tests
// ============================================================================

TEST_F(CVProcessingTest, ComputeIoU_IdenticalBoxes_ReturnsOne) {
  BBox box{0.0f, 0.0f, 10.0f, 10.0f};
  float iou = computeIoU(box, box);
  EXPECT_FLOAT_EQ(iou, 1.0f);
}

TEST_F(CVProcessingTest, ComputeIoU_NoOverlap_ReturnsZero) {
  BBox box1{0.0f, 0.0f, 10.0f, 10.0f};
  BBox box2{20.0f, 20.0f, 30.0f, 30.0f};
  float iou = computeIoU(box1, box2);
  EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST_F(CVProcessingTest, ComputeIoU_PartialOverlap_ReturnsCorrectValue) {
  BBox box1{0.0f, 0.0f, 10.0f, 10.0f}; // Area = 100
  BBox box2{5.0f, 5.0f, 15.0f, 15.0f}; // Area = 100
  // Intersection: (5,5) to (10,10) = 25
  // Union: 100 + 100 - 25 = 175
  // IoU = 25/175 ≈ 0.142857
  float iou = computeIoU(box1, box2);
  EXPECT_NEAR(iou, 0.142857f, 0.0001f);
}

TEST_F(CVProcessingTest, ComputeIoU_OneBoxInsideAnother_ReturnsCorrectValue) {
  BBox box1{0.0f, 0.0f, 10.0f, 10.0f}; // Area = 100
  BBox box2{2.0f, 2.0f, 8.0f, 8.0f};   // Area = 36
  // Intersection: 36 (box2 is fully inside)
  // Union: 100 + 36 - 36 = 100
  // IoU = 36/100 = 0.36
  float iou = computeIoU(box1, box2);
  EXPECT_FLOAT_EQ(iou, 0.36f);
}

// ============================================================================
// BBox Tests
// ============================================================================

TEST_F(CVProcessingTest, BBox_Width_ReturnsCorrectValue) {
  BBox box{0.0f, 0.0f, 10.0f, 5.0f};
  EXPECT_FLOAT_EQ(box.width(), 10.0f);
}

TEST_F(CVProcessingTest, BBox_Height_ReturnsCorrectValue) {
  BBox box{0.0f, 0.0f, 10.0f, 5.0f};
  EXPECT_FLOAT_EQ(box.height(), 5.0f);
}

TEST_F(CVProcessingTest, BBox_Area_ReturnsCorrectValue) {
  BBox box{0.0f, 0.0f, 10.0f, 5.0f};
  EXPECT_FLOAT_EQ(box.area(), 50.0f);
}

TEST_F(CVProcessingTest, BBox_IsValid_ValidBox_ReturnsTrue) {
  BBox box{0.0f, 0.0f, 10.0f, 5.0f};
  EXPECT_TRUE(box.isValid());
}

TEST_F(CVProcessingTest, BBox_IsValid_InvalidBox_ReturnsFalse) {
  BBox box1{10.0f, 0.0f, 5.0f, 5.0f}; // x2 < x1
  EXPECT_FALSE(box1.isValid());

  BBox box2{0.0f, 10.0f, 5.0f, 5.0f}; // y2 < y1
  EXPECT_FALSE(box2.isValid());

  BBox box3{-1.0f, 0.0f, 5.0f, 5.0f}; // negative x1
  EXPECT_FALSE(box3.isValid());
}

TEST_F(CVProcessingTest, BBox_Scale_ReturnsCorrectlyScaledBox) {
  BBox box{1.0f, 2.0f, 3.0f, 4.0f};
  BBox scaled = box.scale(2.0f, 3.0f);
  EXPECT_FLOAT_EQ(scaled.x1, 2.0f);
  EXPECT_FLOAT_EQ(scaled.y1, 6.0f);
  EXPECT_FLOAT_EQ(scaled.x2, 6.0f);
  EXPECT_FLOAT_EQ(scaled.y2, 12.0f);
}

// ============================================================================
// ScaleRatios Tests
// ============================================================================

TEST_F(CVProcessingTest, ScaleRatios_Compute_ReturnsCorrectRatios) {
  cv::Size original(640, 480);
  cv::Size model(320, 240);
  auto ratios = ScaleRatios::compute(original, model);
  EXPECT_FLOAT_EQ(ratios.widthRatio, 2.0f);
  EXPECT_FLOAT_EQ(ratios.heightRatio, 2.0f);
}

// ============================================================================
// validateNormParam Tests
// ============================================================================

TEST_F(CVProcessingTest, ValidateNormParam_ValidThreeElements_ReturnsScalar) {
  std::vector<float> values = {0.5f, 0.6f, 0.7f};
  auto result = validateNormParam(values, "test");
  ASSERT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ((*result)[0], 0.5f);
  EXPECT_FLOAT_EQ((*result)[1], 0.6f);
  EXPECT_FLOAT_EQ((*result)[2], 0.7f);
}

TEST_F(CVProcessingTest, ValidateNormParam_EmptyVector_ReturnsNullopt) {
  std::vector<float> values = {};
  auto result = validateNormParam(values, "test");
  EXPECT_FALSE(result.has_value());
}

TEST_F(CVProcessingTest, ValidateNormParam_WrongSize_ReturnsNullopt) {
  std::vector<float> values = {0.5f, 0.6f}; // Only 2 elements
  auto result = validateNormParam(values, "test");
  EXPECT_FALSE(result.has_value());
}

// ============================================================================
// nonMaxSuppression Tests
// ============================================================================

struct TestDetection {
  BBox bbox;
  float score;
  int32_t classIndex;
};

TEST_F(CVProcessingTest, NonMaxSuppression_EmptyVector_ReturnsEmpty) {
  std::vector<TestDetection> detections = {};
  auto result = nonMaxSuppression(detections, 0.5);
  EXPECT_TRUE(result.empty());
}

TEST_F(CVProcessingTest,
       NonMaxSuppression_SingleDetection_ReturnsSingleDetection) {
  std::vector<TestDetection> detections = {
      {{0.0f, 0.0f, 10.0f, 10.0f}, 0.9f, 1}};
  auto result = nonMaxSuppression(detections, 0.5);
  EXPECT_EQ(result.size(), 1);
  EXPECT_FLOAT_EQ(result[0].score, 0.9f);
}

TEST_F(CVProcessingTest,
       NonMaxSuppression_OverlappingBoxes_SuppressesLowerScore) {
  std::vector<TestDetection> detections = {
      {{0.0f, 0.0f, 10.0f, 10.0f}, 0.9f, 1}, // High score
      {{0.0f, 0.0f, 10.0f, 10.0f}, 0.5f, 1}, // Same box, low score
  };
  auto result = nonMaxSuppression(detections, 0.5);
  EXPECT_EQ(result.size(), 1);
  EXPECT_FLOAT_EQ(result[0].score, 0.9f);
}

TEST_F(CVProcessingTest, NonMaxSuppression_DifferentClasses_KeepsBothBoxes) {
  std::vector<TestDetection> detections = {
      {{0.0f, 0.0f, 10.0f, 10.0f}, 0.9f, 1}, // Class 1
      {{0.0f, 0.0f, 10.0f, 10.0f}, 0.8f, 2}, // Class 2, same location
  };
  auto result = nonMaxSuppression(detections, 0.5);
  EXPECT_EQ(result.size(), 2); // Both should be kept (different classes)
}

TEST_F(CVProcessingTest, NonMaxSuppression_NoOverlap_KeepsAllBoxes) {
  std::vector<TestDetection> detections = {
      {{0.0f, 0.0f, 10.0f, 10.0f}, 0.9f, 1},
      {{20.0f, 20.0f, 30.0f, 30.0f}, 0.8f, 1},
  };
  auto result = nonMaxSuppression(detections, 0.5);
  EXPECT_EQ(result.size(), 2); // Both should be kept (no overlap)
}
