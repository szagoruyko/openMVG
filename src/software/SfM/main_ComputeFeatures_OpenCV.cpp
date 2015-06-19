
// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "openMVG/image/image.hpp"
#include "openMVG/sfm/sfm.hpp"

/// Feature/Regions & Image describer interfaces
#include "openMVG/features/features.hpp"
#include <cereal/archives/json.hpp>

#include "openMVG/system/timer.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/progress/progress.hpp"

/// OpenCV Includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/eigen.hpp"

#include <cstdlib>
#include <fstream>

#include <THC/THC.h>
#include <cunn.h>
#include "loader.h"

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::sfm;
using namespace std;

enum eGeometricModel
{
  FUNDAMENTAL_MATRIX = 0,
  ESSENTIAL_MATRIX   = 1,
  HOMOGRAPHY_MATRIX  = 2
};

enum ePairMode
{
  PAIR_EXHAUSTIVE = 0,
  PAIR_CONTIGUOUS = 1,
  PAIR_FROM_FILE  = 2
};

#define M 64

// Given an image an coordinates+sizes of detected points
// extract corresponding image patches with OpenCV functions
// input image is in [0 255] range,
// the patches are divided by 255 and mean-normalized
// Note: depending on the type of applications you might want to use 
// orientation of detected region or inrease the cropped bounding box by some
// constant
// Another note: this is of course not the fastest way to extracted features.
// The ultimate would be to use CUDA texture memory and process all the features
// from am image in parallel
void extractPatches(const cv::Mat& image,
    const std::vector<cv::KeyPoint>& kp,
    std::vector<cv::Mat>& patches)
{
  for(auto &it : kp)
  {
    cv::Mat patch(M, M, CV_32F);
    cv::Mat buf;
    // increase the size of the region to include some context
    cv::getRectSubPix(image, cv::Size(it.size*1.3, it.size*1.3), it.pt, buf);
    cv::Scalar m = cv::mean(buf);
    cv::resize(buf, patch, cv::Size(M,M));
    patch.convertTo(patch, CV_32F, 1./255.);
    patch = patch.isContinuous() ? patch : patch.clone();
    // mean subtraction is crucial!
    patches.push_back(patch - m[0]/255.);
  }
}

// Copy extracted patches to CUDA memory and run the network
// One has to keep mind that GPU memory is limited and extracting too many patches
// at once might cause troubles
// So if you need to extract a lot of patches, an efficient way would be to
// devide the set in smaller equal parts and preallocate CPU and GPU memory
void extractDescriptors(THCState *state,
    cunn::Sequential::Ptr net,
    const std::vector<cv::Mat>& patches,
    cv::Mat& descriptors)
{
  size_t batch_size = 128;
  size_t N = patches.size();

  THFloatTensor *buffer = THFloatTensor_newWithSize4d(batch_size, 1, M, M);
  THCudaTensor *input = THCudaTensor_newWithSize4d(state, batch_size, 1, M, M);

  for(int j=0; j < ceil((float)N/batch_size); ++j)
  {
    float *data = THFloatTensor_data(buffer);
    size_t k = 0;
    for(size_t i = j*batch_size; i < std::min((j+1)*batch_size, N); ++i, ++k)
      memcpy(data + k*M*M, patches[i].data, sizeof(float) * M * M);

    // initialize 4D CUDA tensor and copy patches into it
    THCudaTensor_copyFloat(state, input, buffer);

    // propagate through the network
    THCudaTensor *output = net->forward(input);

    // copy descriptors back
    THFloatTensor *desc = THFloatTensor_newWithSize2d(output->size[0], output->size[1]);
    THFloatTensor_copyCuda(state, desc, output);

    size_t feature_dim = output->size[1];
    if(descriptors.cols != feature_dim || descriptors.rows != N)
      descriptors.create(N, feature_dim, CV_32F);

    memcpy(descriptors.data + j * feature_dim * batch_size * sizeof(float),
        THFloatTensor_data(desc),
        sizeof(float) * feature_dim * k);

    THFloatTensor_free(desc);
  }

  THCudaTensor_free(state, input);
  THFloatTensor_free(buffer);
}

///
//- Create an Image_describer interface that use an OpenCV feature extraction method
// i.e. with the AKAZE detector+descriptor
//--/!\ If you use a new Regions type you define and register it in
//   "openMVG/features/regions_factory.hpp" file.
///
// Reuse the existing AKAZE floating point Keypoint.
// Define the Interface
class DeepCompare_OCV_Image_describer : public Image_describer
{
public:
  DeepCompare_OCV_Image_describer():Image_describer(){}


  bool Set_configuration_preset(EDESCRIBER_PRESET preset)
  {
    return false;
  }
  /**
  @brief Detect regions on the image and compute their attributes (description)
  @param image Image.
  @param regions The detected regions and attributes (the caller must delete the allocated data)
  @param mask 8-bit gray image for keypoint filtering (optional).
     Non-zero values depict the region of interest.
  */
  bool Describe(const Image<unsigned char>& image,
    std::unique_ptr<Regions> &regions,
    const Image<unsigned char> * mask = NULL)
  {
    THCState *state = (THCState*)malloc(sizeof(THCState));
    THCudaInit(state);

    cv::Mat img;
    cv::eigen2cv(image.GetMat(), img);

    std::vector< cv::KeyPoint > vec_keypoints;
    cv::Mat m_desc;

    //cv::Ptr<cv::Feature2D> extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE);
    //extractor->detectAndCompute(img, cv::Mat(), vec_keypoints, m_desc);
    const char* network_path = "/opt/projects/cvpr15matcher/networks/siam2stream/siam2stream_desc_notredame.bin";
    auto net = loadNetwork(state, network_path);

    //cv::Ptr<cv::MSER> detector = cv::MSER::create(5, 620);
    cv::Ptr<cv::Feature2D> detector = cv::AKAZE::create();
    std::vector<cv::Mat> patches;
    detector->detect(img, vec_keypoints);

    extractPatches(img, vec_keypoints, patches);
    extractDescriptors(state, net, patches, m_desc);

    if (!vec_keypoints.empty())
    {
      Allocate(regions);

      // Build alias to cached data
      DeepCompare_Regions * regionsCasted = dynamic_cast<DeepCompare_Regions*>(regions.get());
      // reserve some memory for faster keypoint saving
      regionsCasted->Features().reserve(vec_keypoints.size());
      regionsCasted->Descriptors().reserve(vec_keypoints.size());

      typedef Descriptor<float, 512> DescriptorT;
      DescriptorT descriptor;
      int cpt = 0;
      for(std::vector< cv::KeyPoint >::const_iterator i_keypoint = vec_keypoints.begin();
        i_keypoint != vec_keypoints.end(); ++i_keypoint, ++cpt){

        SIOPointFeature feat((*i_keypoint).pt.x, (*i_keypoint).pt.y, (*i_keypoint).size, (*i_keypoint).angle);
        regionsCasted->Features().push_back(feat);

        memcpy(descriptor.getData(),
               m_desc.ptr<typename DescriptorT::bin_type>(cpt),
               DescriptorT::static_size*sizeof(typename DescriptorT::bin_type));
        regionsCasted->Descriptors().push_back(descriptor);
      }
    }
    return true;
  };

  /// Allocate Regions type depending of the Image_describer
  void Allocate(std::unique_ptr<Regions> &regions) const
  {
    regions.reset( new DeepCompare_Regions );
  }

  template<class Archive>
  void serialize( Archive & ar )
  {
  }
};
///
//- Create an Image_describer interface that use an OpenCV feature extraction method
// i.e. with the AKAZE detector+descriptor
//--/!\ If you use a new Regions type you define and register it in
//   "openMVG/features/regions_factory.hpp" file.
///
// Reuse the existing AKAZE floating point Keypoint.
typedef features::AKAZE_Float_Regions AKAZE_OpenCV_Regions;
// Define the Interface
class AKAZE_OCV_Image_describer : public Image_describer
{
public:
  AKAZE_OCV_Image_describer():Image_describer(){}


  bool Set_configuration_preset(EDESCRIBER_PRESET preset)
  {
    return false;
  }
  /**
  @brief Detect regions on the image and compute their attributes (description)
  @param image Image.
  @param regions The detected regions and attributes (the caller must delete the allocated data)
  @param mask 8-bit gray image for keypoint filtering (optional).
     Non-zero values depict the region of interest.
  */
  bool Describe(const Image<unsigned char>& image,
    std::unique_ptr<Regions> &regions,
    const Image<unsigned char> * mask = NULL)
  {
    cv::Mat img;
    cv::eigen2cv(image.GetMat(), img);

    std::vector< cv::KeyPoint > vec_keypoints;
    cv::Mat m_desc;

    cv::Ptr<cv::Feature2D> extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE);
    extractor->detectAndCompute(img, cv::Mat(), vec_keypoints, m_desc);

    if (!vec_keypoints.empty())
    {
      Allocate(regions);

      // Build alias to cached data
      AKAZE_OpenCV_Regions * regionsCasted = dynamic_cast<AKAZE_OpenCV_Regions*>(regions.get());
      // reserve some memory for faster keypoint saving
      regionsCasted->Features().reserve(vec_keypoints.size());
      regionsCasted->Descriptors().reserve(vec_keypoints.size());

      typedef Descriptor<float, 64> DescriptorT;
      DescriptorT descriptor;
      int cpt = 0;
      for(std::vector< cv::KeyPoint >::const_iterator i_keypoint = vec_keypoints.begin();
        i_keypoint != vec_keypoints.end(); ++i_keypoint, ++cpt){

        SIOPointFeature feat((*i_keypoint).pt.x, (*i_keypoint).pt.y, (*i_keypoint).size, (*i_keypoint).angle);
        regionsCasted->Features().push_back(feat);

        memcpy(descriptor.getData(),
               m_desc.ptr<typename DescriptorT::bin_type>(cpt),
               DescriptorT::static_size*sizeof(typename DescriptorT::bin_type));
        regionsCasted->Descriptors().push_back(descriptor);
      }
    }
    return true;
  };

  /// Allocate Regions type depending of the Image_describer
  void Allocate(std::unique_ptr<Regions> &regions) const
  {
    regions.reset( new AKAZE_OpenCV_Regions );
  }

  template<class Archive>
  void serialize( Archive & ar )
  {
  }
};
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>
CEREAL_REGISTER_TYPE_WITH_NAME(AKAZE_OCV_Image_describer, "AKAZE_OCV_Image_describer");
CEREAL_REGISTER_TYPE_WITH_NAME(DeepCompare_OCV_Image_describer, "DeepCompare_OCV_Image_describer");

/// Compute between the Views
/// Compute view image description (feature & descriptor extraction using OpenCV)
/// Compute putative local feature matches (descriptor matching)
/// Compute geometric coherent feature matches (robust model estimation from putative matches)
/// Export computed data
int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutDir = "";
  bool bForce = false;

  // required
  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add( make_option('o', sOutDir, "outdir") );
  // Optional
  cmd.add( make_option('f', bForce, "force") );

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-i|--input_file]: a SfM_Data file \n"
      << "[-o|--outdir path] \n"
      << "\n[Optional]\n"
      << "[-f|--force: Force to recompute data]\n"
      << std::endl;

      std::cerr << s << std::endl;
      return EXIT_FAILURE;
  }

  std::cout << " You called : " <<std::endl
            << argv[0] << std::endl
            << "--input_file " << sSfM_Data_Filename << std::endl
            << "--outdir " << sOutDir << std::endl;

  if (sOutDir.empty())  {
    std::cerr << "\nIt is an invalid output directory" << std::endl;
    return EXIT_FAILURE;
  }

  // Create output dir
  if (!stlplus::folder_exists(sOutDir))
  {
    if (!stlplus::folder_create(sOutDir))
    {
      std::cerr << "Cannot create output directory" << std::endl;
      return EXIT_FAILURE;
    }
  }

  //---------------------------------------
  // a. Load input scene
  //---------------------------------------
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS))) {
    std::cerr << std::endl
      << "The input file \""<< sSfM_Data_Filename << "\" cannot be read" << std::endl;
    return false;
  }

  // Init the image_describer
  // - retrieve the used one in case of pre-computed features
  // - else create the desired one

  using namespace openMVG::features;
  std::unique_ptr<Image_describer> image_describer;

  const std::string sImage_describer = stlplus::create_filespec(sOutDir, "image_describer", "json");
  if (stlplus::is_file(sImage_describer))
  {
    // Dynamically load the image_describer from the file (will restore old used settings)
    std::ifstream stream(sImage_describer.c_str());
    if (!stream.is_open())
      return false;

    cereal::JSONInputArchive archive(stream);
    archive(cereal::make_nvp("image_describer", image_describer));
  }
  else
  {
    //image_describer.reset(new AKAZE_OCV_Image_describer);
    image_describer.reset(new DeepCompare_OCV_Image_describer);

    // Export the used Image_describer and region type for:
    // - dynamic future regions computation and/or loading
    {
      std::ofstream stream(sImage_describer.c_str());
      if (!stream.is_open())
        return false;

      cereal::JSONOutputArchive archive(stream);
      archive(cereal::make_nvp("image_describer", image_describer));
      std::unique_ptr<Regions> regionsType;
      image_describer->Allocate(regionsType);
      archive(cereal::make_nvp("regions_type", regionsType));
    }
  }

  // Feature extraction routines
  // For each View of the SfM_Data container:
  // - if regions file exist continue,
  // - if no file, compute features
  {
    system::Timer timer;
    Image<unsigned char> imageGray;
    C_Progress_display my_progress_bar( sfm_data.GetViews().size(),
      std::cout, "\n- EXTRACT FEATURES -\n" );
    for(Views::const_iterator iterViews = sfm_data.views.begin();
        iterViews != sfm_data.views.end();
        ++iterViews, ++my_progress_bar)
    {
      const View * view = iterViews->second.get();
      const std::string sView_filename = stlplus::create_filespec(sfm_data.s_root_path,
        view->s_Img_path);
      const std::string sFeat = stlplus::create_filespec(sOutDir,
        stlplus::basename_part(sView_filename), "feat");
      const std::string sDesc = stlplus::create_filespec(sOutDir,
        stlplus::basename_part(sView_filename), "desc");

      //If features or descriptors file are missing, compute them
      if (bForce || !stlplus::file_exists(sFeat) || !stlplus::file_exists(sDesc))
      {
        if (!ReadImage(sView_filename.c_str(), &imageGray))
          continue;

        // Compute features and descriptors and export them to files
        std::unique_ptr<Regions> regions;
        image_describer->Describe(imageGray, regions);
        image_describer->Save(regions.get(), sFeat, sDesc);
      }
    }
    std::cout << "Task done in (s): " << timer.elapsed() << std::endl;
  }
  return EXIT_SUCCESS;
}
