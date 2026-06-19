#ifndef HDF5_WRITER_H
#define HDF5_WRITER_H

#include "IDataWriter.h"
#include "HDF5WriteConfig.h"
#include "FragmentPair.h"
#include <H5Cpp.h>
#include <string>
#include <vector>

class HDF5Writer : public IDataWriter {
private:
  std::string filepath_;
  HDF5WriteConfig config_;
  
  H5::H5File* file_;
    
  // Buffers
  std::vector<int> A_light_;
  std::vector<int> Z_light_;
  std::vector<double> KE_light_;
  std::vector<int> n_light_;
  std::vector<double> I40_light_;
  std::vector<double> Init_light_;
  /*std::vector<double> ME0_light_;
  std::vector<double> ME1_light_;
  std::vector<double> ME2_light_;
  std::vector<double> ME3_light_;
  std::vector<double> ME4_light_;
  std::vector<double> MZ1_light_;
  std::vector<double> MZ2_light_;
  std::vector<double> MZ3_light_;
  std::vector<double> MZ4_light_;

  std::vector<double> MZE_light_;
  std::vector<double> MZ2E_light_;
  std::vector<double> MZE2_light_;
  
  std::vector<double> MN2_light_;
  std::vector<double> MN3_light_;
  std::vector<double> MN4_light_;
  std::vector<double> Rt_light_;
  std::vector<double> R_light_;
  std::vector<double> pD1_light_;
  std::vector<double> pD2_light_;
*/
  std::vector<int> A_heavy_;
  std::vector<int> Z_heavy_;
  std::vector<double> KE_heavy_;
  std::vector<int> n_heavy_;
  std::vector<double> I40_heavy_;
  std::vector<double> Init_heavy_;
  /*std::vector<double> ME0_heavy_;
  std::vector<double> ME1_heavy_;
  std::vector<double> ME2_heavy_;
  std::vector<double> ME3_heavy_;
  std::vector<double> ME4_heavy_;
  std::vector<double> MZ1_heavy_;
  std::vector<double> MZ2_heavy_;
  std::vector<double> MZ3_heavy_;
  std::vector<double> MZ4_heavy_;

  std::vector<double> MZE_heavy_;
  std::vector<double> MZ2E_heavy_;
  std::vector<double> MZE2_heavy_;
 
  std::vector<double> Rt_heavy_;
  std::vector<double> MN2_heavy_;
  std::vector<double> MN3_heavy_;
  std::vector<double> MN4_heavy_;
  std::vector<double> R_heavy_;
  std::vector<double> pD1_heavy_;
  std::vector<double> pD2_heavy_;

*/
  std::vector<std::vector<double>> betheTrace_light_;
  std::vector<std::vector<double>>rawTrace_light_;
  std::vector<std::vector<double>>denoisedTrace_light_;
  
  std::vector<std::vector<double>>betheTrace_heavy_;
  std::vector<std::vector<double>>rawTrace_heavy_;
  std::vector<std::vector<double>>denoisedTrace_heavy_;

  size_t bufferSize_;
  const size_t MAX_BUFFER_SIZE = 1000;
    
public:
  explicit HDF5Writer(const std::string& filepath, 
		      const HDF5WriteConfig& config = HDF5WriteConfig());
  ~HDF5Writer();
  
  void writeFragmentPair(const FragmentPair& pair, const int event_ID) override;
  void writeDenoisingPair(const FragmentPair& pair) override;
  
  void finalize() override;
  
  void addMetadata(const std::string& key, double value);
  void addMetadata(const std::string& key, const std::string& value);
  
private:
  void initialize();
  void flushBuffers();

  size_t denoising_trace_count_ = 0;
  size_t max_trace_length_seen_ = 0;
  
};

#endif

