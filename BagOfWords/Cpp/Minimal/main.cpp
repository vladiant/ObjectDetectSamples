// https://raw.githubusercontent.com/opencv/opencv_contrib/a26f71313009c93d105151094436eecd4a0990ed/modules/xfeatures2d/samples/bagofwords_classification.cpp
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>

constexpr int memoryUse = 200;
constexpr int vocabSize = 1000;
constexpr float descProportion = 0.3f;

constexpr bool balanceClasses = true;

class ObdImage {
 public:
  ObdImage(std::string p_id, std::string p_path) : id(p_id), path(p_path) {}
  std::string id;
  std::string path;
};

cv::Mat trainVocabulary(
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const cv::Ptr<cv::DescriptorExtractor>& dextractor,
    const std::map<std::string, std::vector<std::filesystem::path>>&
        classImages) {
  cv::Mat vocabulary;

  CV_Assert(dextractor->descriptorType() == CV_32FC1);
  const int elemSize = CV_ELEM_SIZE(dextractor->descriptorType());
  const int descByteSize = dextractor->descriptorSize() * elemSize;
  const int bytesInMB = 1048576;
  const int maxDescCount =
      (memoryUse * bytesInMB) /
      descByteSize;  // Total number of descs to use for training.

  std::vector<ObdImage> images;
  for (const auto& imageData : classImages) {
    for (const auto& imagePath : imageData.second) {
      images.emplace_back(imageData.first, imagePath.string());
    }
  }

  std::cout << "Computing descriptors...\n";
  cv::RNG& rng = cv::theRNG();
  cv::TermCriteria terminate_criterion;
  terminate_criterion.epsilon = FLT_EPSILON;
  cv::BOWKMeansTrainer bowTrainer(vocabSize, terminate_criterion, 3,
                                  cv::KmeansFlags::KMEANS_PP_CENTERS);

  while (images.size() > 0) {
    if (bowTrainer.descriptorsCount() > maxDescCount) {
      break;
    }

    // Randomly pick an image from the dataset which hasn't yet been seen
    // and compute the descriptors from that image.
    int randImgIdx = rng((unsigned)images.size());
    cv::Mat colorImage = cv::imread(images[randImgIdx].path);
    std::vector<cv::KeyPoint> imageKeypoints;
    fdetector->detect(colorImage, imageKeypoints);
    cv::Mat imageDescriptors;
    dextractor->compute(colorImage, imageKeypoints, imageDescriptors);

    // check that there were descriptors calculated for the current image
    if (!imageDescriptors.empty()) {
      int descCount = imageDescriptors.rows;
      // Extract descProportion descriptors from the image,
      // breaking if the 'allDescriptors' matrix becomes full
      int descsToExtract =
          static_cast<int>(descProportion * static_cast<float>(descCount));
      // Fill mask of used descriptors
      std::vector<char> usedMask(descCount, false);
      fill(usedMask.begin(), usedMask.begin() + descsToExtract, true);
      for (int i = 0; i < descCount; i++) {
        int i1 = rng(descCount), i2 = rng(descCount);
        char tmp = usedMask[i1];
        usedMask[i1] = usedMask[i2];
        usedMask[i2] = tmp;
      }

      for (int i = 0; i < descCount; i++) {
        if (usedMask[i] && bowTrainer.descriptorsCount() < maxDescCount)
          bowTrainer.add(imageDescriptors.row(i));
      }
    }

    // Delete the current element from images so it is not added again
    images.erase(images.begin() + randImgIdx);
  }

  std::cout << "Maximum allowed descriptor count: " << maxDescCount
            << ", Actual descriptor count: " << bowTrainer.descriptorsCount()
            << '\n';

  std::cout << "Training vocabulary...\n";
  vocabulary = bowTrainer.cluster();

  return vocabulary;
}

void calculateImageDescriptors(
    const std::vector<ObdImage>& images, std::vector<cv::Mat>& imageDescriptors,
    const cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor,
    const cv::Ptr<cv::FeatureDetector>& fdetector) {
  CV_Assert(!bowExtractor->getVocabulary().empty());
  imageDescriptors.resize(images.size());

  for (size_t i = 0; i < images.size(); i++) {
    cv::Mat colorImage = cv::imread(images[i].path);
    std::vector<cv::KeyPoint> keypoints;
    fdetector->detect(colorImage, keypoints);
    bowExtractor->compute(colorImage, keypoints, imageDescriptors[i]);
  }
}

void removeEmptyBowImageDescriptors(std::vector<ObdImage>& images,
                                    std::vector<cv::Mat>& bowImageDescriptors,
                                    std::vector<char>& objectPresent) {
  CV_Assert(!images.empty());
  for (int i = (int)images.size() - 1; i >= 0; i--) {
    bool res = bowImageDescriptors[i].empty();
    if (res) {
      images.erase(images.begin() + i);
      bowImageDescriptors.erase(bowImageDescriptors.begin() + i);
      objectPresent.erase(objectPresent.begin() + i);
    }
  }
}

void setSVMParams(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& responses,
                  bool balanceClasses) {
  int pos_ex = cv::countNonZero(responses == 1);
  int neg_ex = cv::countNonZero(responses == -1);
  std::cout << pos_ex << " positive training samples; " << neg_ex
            << " negative training samples\n";

  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::RBF);
  if (balanceClasses) {
    cv::Mat class_wts(2, 1, CV_32FC1);
    // The first training sample determines the '+1' class internally, even if
    // it is negative, so store whether this is the case so that the class
    // weights can be reversed accordingly.
    bool reversed_classes = (responses.at<float>(0) < 0.f);
    if (reversed_classes == false) {
      class_wts.at<float>(0) =
          static_cast<float>(pos_ex) /
          static_cast<float>(
              pos_ex +
              neg_ex);  // weighting for costs of positive class + 1 (i.e. cost
                        // of false positive - larger gives greater cost)
      class_wts.at<float>(1) =
          static_cast<float>(neg_ex) /
          static_cast<float>(pos_ex +
                             neg_ex);  // weighting for costs of negative class
                                       // - 1 (i.e. cost of false negative)
    } else {
      class_wts.at<float>(0) =
          static_cast<float>(neg_ex) / static_cast<float>(pos_ex + neg_ex);
      class_wts.at<float>(1) =
          static_cast<float>(pos_ex) / static_cast<float>(pos_ex + neg_ex);
    }
    svm->setClassWeights(class_wts);
  }
}

cv::Ptr<cv::ml::SVM> trainSVMClassifier(
    const std::string& objClassName,
    const cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor,
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const std::map<std::string, std::vector<std::filesystem::path>>&
        classImages) {
  std::cout << "*** TRAINING CLASSIFIER FOR CLASS " << objClassName << " ***\n";
  std::cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName
            << "...\n";

  // Get classification ground truth for images in the training set
  std::vector<ObdImage> images;
  std::vector<char> objectPresent;
  for (const auto& imageData : classImages) {
    for (const auto& imagePath : imageData.second) {
      images.emplace_back(imageData.first, imagePath.string());
      objectPresent.push_back(imageData.first == objClassName);
    }
  }

  // Compute the bag of words vector for each image in the training set.
  std::vector<cv::Mat> bowImageDescriptors;
  calculateImageDescriptors(images, bowImageDescriptors, bowExtractor,
                            fdetector);

  // Remove any images for which descriptors could not be calculated
  removeEmptyBowImageDescriptors(images, bowImageDescriptors, objectPresent);

  // Prepare the input matrices for SVM training.
  cv::Mat trainData((int)images.size(), bowExtractor->getVocabulary().rows,
                    CV_32FC1);
  cv::Mat responses((int)images.size(), 1, CV_32SC1);

  // Transfer bag of words vectors and responses across to the training data
  // matrices
  for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++) {
    // Transfer image descriptor (bag of words vector) to training data matrix
    cv::Mat submat = trainData.row((int)imageIdx);
    if (bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize()) {
      std::cout << "Error: computed bow image descriptor size "
                << bowImageDescriptors[imageIdx].cols
                << " differs from vocabulary size "
                << bowExtractor->getVocabulary().cols << std::endl;
      exit(EXIT_FAILURE);
    }
    bowImageDescriptors[imageIdx].copyTo(submat);

    // Set response value
    responses.at<int>((int)imageIdx) = objectPresent[imageIdx] ? 1 : -1;
  }

  std::cout << "TRAINING SVM FOR CLASS ..." << objClassName << "...\n";
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  setSVMParams(svm, responses, balanceClasses);

  // Training finding the best parameters (grid can be adjusted)
  svm->trainAuto(
      cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, responses));

  std::cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED\n";

  return svm;
}

void computeConfidences(
    const cv::Ptr<cv::ml::SVM>& svm, const std::string& objClassName,
    cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor,
    const cv::Ptr<cv::FeatureDetector>& fdetector,
    const std::map<std::string, std::vector<std::filesystem::path>>&
        testImages) {
  std::cout << "*** CALCULATING CONFIDENCES FOR CLASS " << objClassName
            << " ***\n";
  std::cout << "CALCULATING BOW VECTORS FOR TEST SET OF " << objClassName
            << "...\n";
  // Get classification ground truth for images in the test set
  std::vector<ObdImage> images;
  std::vector<cv::Mat> bowImageDescriptors;
  std::vector<char> objectPresent;
  for (const auto& imageData : testImages) {
    for (const auto& imagePath : imageData.second) {
      images.emplace_back(imageData.first, imagePath.string());
      objectPresent.push_back(imageData.first == objClassName);
    }
  }

  // Compute the bag of words vector for each image in the test set
  calculateImageDescriptors(images, bowImageDescriptors, bowExtractor,
                            fdetector);
  // Remove any images for which descriptors could not be calculated
  removeEmptyBowImageDescriptors(images, bowImageDescriptors, objectPresent);

  // Use the bag of words vectors to calculate classifier output for each image
  // in test set
  std::cout << "CALCULATING CONFIDENCE SCORES FOR CLASS " << objClassName
            << "...\n";
  std::vector<float> confidences(images.size());
  float signMul = 1.f;
  size_t falseNegativeCount = 0;
  size_t positiveCount = 0;
  size_t falsePositiveCount = 0;
  size_t negativeCount = 0;
  for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++) {
    if (imageIdx == 0) {
      // In the first iteration, determine the sign of the positive class
      float classVal = confidences[imageIdx] =
          svm->predict(bowImageDescriptors[imageIdx], cv::noArray(), 0);
      float scoreVal = confidences[imageIdx] =
          svm->predict(bowImageDescriptors[imageIdx], cv::noArray(),
                       cv::ml::StatModel::RAW_OUTPUT);
      signMul = (classVal < 0) == (scoreVal < 0) ? 1.f : -1.f;
    }
    // svm output of decision function
    confidences[imageIdx] =
        signMul * svm->predict(bowImageDescriptors[imageIdx], cv::noArray(),
                               cv::ml::StatModel::RAW_OUTPUT);

    if (images[imageIdx].id == objClassName) {
      positiveCount++;
      if (confidences[imageIdx] < 0) {
        falseNegativeCount++;
      }
    } else {
      negativeCount++;
      if (confidences[imageIdx] >= 0) {
        falsePositiveCount++;
      }
    }

    std::cout << images[imageIdx].path
              << " expected class: " << images[imageIdx].id
              << " confidence for class " << objClassName << " "
              << confidences[imageIdx] << std::endl;
  }

  std::cout << " CLASS " << objClassName << " :"
            << " false negative: " << falseNegativeCount
            << " positive: " << positiveCount - falseNegativeCount
            << " false positive: " << falsePositiveCount
            << " negative: " << negativeCount - falsePositiveCount << '\n';
}

int main(int argc, char** argv) {
  cv::String keys =
      "{help h usage ? | | print this message   }"
      "{t trainingSet  | <none> | path to training set  }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Program to train Bag of Words feature based classifier");

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  if (!parser.has("trainingSet")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  if (!parser.check()) {
    parser.printErrors();
    return EXIT_SUCCESS;
  }

  const cv::String trainSetPath = parser.get<cv::String>("trainingSet");

  // Read image class names from subfolders
  std::cout << "Reading class data...\n";
  std::map<std::string, std::filesystem::path> classPaths;
  for (const auto& dirEntry :
       std::filesystem::directory_iterator{trainSetPath}) {
    const std::filesystem::path localPath = dirEntry;
    if (dirEntry.is_directory()) {
      classPaths[dirEntry.path().filename()] = dirEntry.path();
    }
  }

  // Read image filenames for each class
  std::map<std::string, std::vector<std::filesystem::path>> classImages;
  for (const auto& classData : classPaths) {
    for (const auto& imagePath :
         std::filesystem::directory_iterator{classData.second}) {
      classImages[classData.first].push_back(imagePath);
    }
  }

  // Create detector, descriptor, matcher.

  // Note that combined detector/descriptor should be used
  // with float type descriptors
  // The only suitable is KAZE
  cv::Ptr<cv::Feature2D> featureDetector = cv::KAZE::create();
  cv::Ptr<cv::DescriptorExtractor> descExtractor = featureDetector;
  cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor;

  if (!featureDetector || !descExtractor) {
    std::cout << "featureDetector or descExtractor was not created\n";
    return EXIT_FAILURE;
  }

  {
    cv::Ptr<cv::DescriptorMatcher> descMatcher =
        cv::DescriptorMatcher::create("FlannBased");
    if (!descMatcher) {
      std::cout << "descMatcher was not created\n";
      return EXIT_FAILURE;
    }
    bowExtractor =
        cv::makePtr<cv::BOWImgDescriptorExtractor>(descExtractor, descMatcher);
  }

  // 1. Train visual word vocabulary
  cv::Mat vocabulary =
      trainVocabulary(featureDetector, descExtractor, classImages);
  bowExtractor->setVocabulary(vocabulary);

  // 2. Train a classifier and run a sample query for each object class
  const auto objClasses = [&classImages]() {
    std::vector<std::string> result;
    for (const auto& imageData : classImages) {
      result.push_back(imageData.first);
    }
    return result;
  }();

  for (size_t classIdx = 0; classIdx < objClasses.size(); ++classIdx) {
    // Train a classifier on train dataset
    cv::Ptr<cv::ml::SVM> svm = trainSVMClassifier(
        objClasses[classIdx], bowExtractor, featureDetector, classImages);

    // Now use the classifier over all images on the test dataset and rank
    // according to score order also calculating precision-recall etc.
    computeConfidences(svm, objClasses[classIdx], bowExtractor, featureDetector,
                       classImages);  // test set here
  }

  return EXIT_SUCCESS;
}
