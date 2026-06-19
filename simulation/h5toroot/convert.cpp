#include <H5Cpp.h>
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <set>
#include <algorithm>

// Structure pour stocker info sur un dataset
struct DatasetInfo {
    std::string name;
    bool isVLen;
    H5::DataType dtype;
    hsize_t size;
};

// Structure pour un groupe avec ses datasets
struct GroupData {
    std::string fullPath;
    std::string branchPrefix;
    std::vector<DatasetInfo> datasets;
};

// Progress bar
void printProgress(size_t current, size_t total) {
    int barWidth = 50;
    float progress = (float)current / total;
    int pos = barWidth * progress;
    
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << ")\r";
    std::cout.flush();
}

// Convertir path HDF5 en préfixe branch ROOT
std::string pathToBranchPrefix(const std::string& path) {
    if (path == "/") return "";
    
    std::string prefix = path;
    // Enlever "/" initial
    if (prefix[0] == '/') prefix = prefix.substr(1);
    // Remplacer "/" par "_"
    std::replace(prefix.begin(), prefix.end(), '/', '_');
    // Ajouter "_" final
    if (!prefix.empty()) prefix += "_";
    
    return prefix;
}

// Scanner récursivement tous les groupes (max depth=3)
void getAllGroupsRecursive(H5::Group& group, const std::string& currentPath, 
                          std::vector<std::string>& groups, int depth, int maxDepth) {
    if (depth > maxDepth) return;
    
    for (hsize_t i = 0; i < group.getNumObjs(); i++) {
        std::string objName = group.getObjnameByIdx(i);
        H5G_obj_t objType = group.getObjTypeByIdx(i);
        
        if (objType == H5G_GROUP) {
            std::string subPath = currentPath + "/" + objName;
            groups.push_back(subPath);
            
            // Récursion
            H5::Group subgroup = group.openGroup(objName);
            getAllGroupsRecursive(subgroup, subPath, groups, depth + 1, maxDepth);
        }
    }
}

std::vector<std::string> getAllGroups(H5::H5File& file, int maxDepth = 3) {
    std::vector<std::string> groups;
    H5::Group root = file.openGroup("/");
    getAllGroupsRecursive(root, "", groups, 0, maxDepth);
    
    // Tri alphabétique
    std::sort(groups.begin(), groups.end());
    
    return groups;
}

// Scanner un groupe HDF5
std::vector<DatasetInfo> scanGroup(H5::Group& group) {
    std::vector<DatasetInfo> datasets;
    
    for (hsize_t i = 0; i < group.getNumObjs(); i++) {
        std::string objName = group.getObjnameByIdx(i);
        H5G_obj_t objType = group.getObjTypeByIdx(i);
        
        if (objType == H5G_DATASET) {
            H5::DataSet ds = group.openDataSet(objName);
            H5::DataSpace space = ds.getSpace();
            H5::DataType dtype = ds.getDataType();
            
            DatasetInfo info;
            info.name = objName;
            info.dtype = dtype;
            info.size = space.getSimpleExtentNpoints();
            info.isVLen = (dtype.getClass() == H5T_VLEN);
            
            datasets.push_back(info);
            
            std::cout << "  Found: " << objName 
                      << (info.isVLen ? " (VLen)" : " (Fixed)")
                      << " [" << info.size << " entries]" << std::endl;
        }
    }
    
    return datasets;
}

// Lire dataset VLen
std::vector<std::vector<float>> readVLenDataset(H5::DataSet& ds, size_t nEvents) {
    std::vector<std::vector<float>> data(nEvents);
    
    hvl_t* rdata = new hvl_t[nEvents];
    H5::VarLenType vlType(&H5::PredType::NATIVE_FLOAT);
    ds.read(rdata, vlType);
    
    for (size_t i = 0; i < nEvents; i++) {
        data[i].resize(rdata[i].len);
        float* ptr = (float*)rdata[i].p;
        for (size_t j = 0; j < rdata[i].len; j++) {
            data[i][j] = ptr[j];
        }
    }
    
    H5::DataSet::vlenReclaim(rdata, vlType, H5::DataSpace(H5S_SCALAR));
    delete[] rdata;
    
    return data;
}

// Conversion principale
void convertH5ToRoot(const std::string& h5file, const std::string& rootfile, int maxlen) {
    std::cout << "Opening HDF5 file: " << h5file << std::endl;
    H5::H5File file(h5file, H5F_ACC_RDONLY);
    
    // Scanner tous les groupes (sauf metadata)
    std::cout << "\n=== Scanning HDF5 structure ===" << std::endl;
    std::vector<std::string> allGroupPaths = getAllGroups(file);
    
    std::vector<GroupData> allGroups;
    size_t nEvents = 0;
    
    for (const auto& groupPath : allGroupPaths) {
        // Ignorer metadata
        if (groupPath.find("metadata") != std::string::npos) {
            std::cout << "Skipping: " << groupPath << " (metadata)" << std::endl;
            continue;
        }
        
        std::cout << "\n=== Scanning " << groupPath << " ===" << std::endl;
        H5::Group group = file.openGroup(groupPath);
        auto datasets = scanGroup(group);
        
        if (datasets.empty()) {
            std::cout << "  (no datasets)" << std::endl;
            continue;
        }
        
        GroupData gdata;
        gdata.fullPath = groupPath;
        gdata.branchPrefix = pathToBranchPrefix(groupPath);
        gdata.datasets = datasets;
        allGroups.push_back(gdata);
        
        // Nombre d'événements (du premier dataset trouvé)
        if (nEvents == 0) {
            nEvents = datasets[0].size;
        }
    }
    
    std::cout << "\nTotal events: " << nEvents << std::endl;
    std::cout << "MAXLEN: " << maxlen << std::endl;
    
    // Vérifier collisions noms de branches
    std::set<std::string> branchNames;
    for (const auto& gdata : allGroups) {
        for (const auto& info : gdata.datasets) {
            std::string branchName = gdata.branchPrefix + info.name;
            if (branchNames.count(branchName)) {
                std::cerr << "\nERROR: Branch name collision detected: '" << branchName << "'" << std::endl;
                std::cerr << "Please rename groups in HDF5 to avoid conflicts." << std::endl;
                std::cerr << "Example: /" << gdata.fullPath << "/" << info.name << std::endl;
                throw std::runtime_error("Branch name collision");
            }
            branchNames.insert(branchName);
        }
    }
    
    // Créer ROOT file
    std::cout << "\nCreating ROOT file: " << rootfile << std::endl;
    TFile* fout = new TFile(rootfile.c_str(), "RECREATE");
    TTree* tree = new TTree("T2F", "Converted from HDF5");
    tree->SetAutoFlush(10000);
    
    // Variables pour branches
    std::map<std::string, int> intVars;
    std::map<std::string, float> floatVars;
    std::map<std::string, float*> arrayVars;
    
    int truncateCount = 0;
    
    // Créer branches pour tous les groupes
    for (const auto& gdata : allGroups) {
        H5::Group group = file.openGroup(gdata.fullPath);
        
        for (const auto& info : gdata.datasets) {
            std::string branchName = gdata.branchPrefix + info.name;
            
            if (info.isVLen) {
                float* arr = new float[maxlen];
                arrayVars[branchName] = arr;
                tree->Branch(branchName.c_str(), arr, 
                           (branchName + "[" + std::to_string(maxlen) + "]/F").c_str());
            } else {
                H5::DataSet ds = group.openDataSet(info.name);
                H5T_class_t tclass = info.dtype.getClass();
                
                if (tclass == H5T_INTEGER) {
                    intVars[branchName] = 0;
                    tree->Branch(branchName.c_str(), &intVars[branchName], 
                               (branchName + "/I").c_str());
                } else if (tclass == H5T_FLOAT) {
                    floatVars[branchName] = 0.0f;
                    tree->Branch(branchName.c_str(), &floatVars[branchName], 
                               (branchName + "/F").c_str());
                }
            }
        }
    }
    
    // Lire tous les datasets en mémoire
    std::cout << "\nReading HDF5 data..." << std::endl;
    
    std::map<std::string, std::vector<int>> intData;
    std::map<std::string, std::vector<float>> floatData;
    std::map<std::string, std::vector<std::vector<float>>> vlenData;
    
    for (const auto& gdata : allGroups) {
        H5::Group group = file.openGroup(gdata.fullPath);
        
        for (const auto& info : gdata.datasets) {
            H5::DataSet ds = group.openDataSet(info.name);
            std::string key = gdata.branchPrefix + info.name;
            
            if (info.isVLen) {
                vlenData[key] = readVLenDataset(ds, nEvents);
            } else {
                H5T_class_t tclass = info.dtype.getClass();
                if (tclass == H5T_INTEGER) {
                    intData[key].resize(nEvents);
                    ds.read(intData[key].data(), H5::PredType::NATIVE_INT);
                } else if (tclass == H5T_FLOAT) {
                    floatData[key].resize(nEvents);
                    ds.read(floatData[key].data(), H5::PredType::NATIVE_FLOAT);
                }
            }
        }
    }
    
    // Remplir TTree
    std::cout << "\nFilling TTree..." << std::endl;
    for (size_t i = 0; i < nEvents; i++) {
        // Scalaires
        for (auto& pair : intVars) {
            pair.second = intData[pair.first][i];
        }
        for (auto& pair : floatVars) {
            pair.second = floatData[pair.first][i];
        }
        
        // Arrays VLen
        for (auto& pair : arrayVars) {
            auto& trace = vlenData[pair.first][i];
            size_t len = trace.size();
            
            if (len > (size_t)maxlen) {
                if (truncateCount == 0) {
                    std::cout << "\nWARNING: Trace truncated at event " << i 
                              << " (" << pair.first << " len=" << len << " > " << maxlen << ")" << std::endl;
                }
                truncateCount++;
                len = maxlen;
            }
            
            for (size_t j = 0; j < len; j++) {
                pair.second[j] = trace[j];
            }
            for (size_t j = len; j < (size_t)maxlen; j++) {
                pair.second[j] = 0.0f;
            }
        }
        
        tree->Fill();
        
        if (i % 1000 == 0) {
            printProgress(i, nEvents);
        }
    }
    printProgress(nEvents, nEvents);
    std::cout << std::endl;
    
    if (truncateCount > 0) {
        std::cout << "\nTotal truncated traces: " << truncateCount << std::endl;
    }
    
    // Copier métadonnées
    if (file.nameExists("/metadata")) {
        std::cout << "\nCopying metadata..." << std::endl;
        H5::Group metaGroup = file.openGroup("/metadata");
        
        for (hsize_t i = 0; i < metaGroup.getNumObjs(); i++) {
            std::string objName = metaGroup.getObjnameByIdx(i);
            H5G_obj_t objType = metaGroup.getObjTypeByIdx(i);
            
            if (objType == H5G_DATASET) {
                H5::DataSet ds = metaGroup.openDataSet(objName);
                H5::DataType dtype = ds.getDataType();
                
                if (dtype.getClass() == H5T_STRING) {
                    std::string value;
                    ds.read(value, dtype);
                    TNamed* obj = new TNamed(objName.c_str(), value.c_str());
                    obj->Write();
                    std::cout << "  " << objName << " = " << value << std::endl;
                }
            }
        }
    }
    
    // Écrire et fermer
    std::cout << "\nWriting ROOT file..." << std::endl;
    tree->Write();
    fout->Close();
    delete fout;
    
    // Libérer mémoire
    for (auto& pair : arrayVars) {
        delete[] pair.second;
    }
    
    std::cout << "\nConversion complete!" << std::endl;
}

// Validation
void validateConversion(const std::string& h5file, const std::string& rootfile, int nCheck = 100) {
    std::cout << "\n=== Validation ===" << std::endl;
    std::cout << "Checking first " << nCheck << " events..." << std::endl;
    
    // Trouver premier groupe avec datasets
    H5::H5File h5f(h5file, H5F_ACC_RDONLY);
    std::vector<std::string> groups = getAllGroups(h5f);
    
    if (groups.empty()) {
        std::cout << "No groups found for validation" << std::endl;
        return;
    }
    
    // Utiliser premier groupe non-metadata
    std::string testGroup;
    for (const auto& g : groups) {
        if (g.find("metadata") == std::string::npos) {
            testGroup = g;
            break;
        }
    }
    
    if (testGroup.empty()) {
        std::cout << "No valid groups for validation" << std::endl;
        return;
    }
    
    std::cout << "Using group: " << testGroup << std::endl;
    
    H5::Group group = h5f.openGroup(testGroup);
    auto datasets = scanGroup(group);
    
    if (datasets.empty()) {
        std::cout << "No datasets in group" << std::endl;
        return;
    }
    
    // Prendre premiers datasets int
    std::string ds1Name, ds2Name;
    for (const auto& info : datasets) {
        if (!info.isVLen && info.dtype.getClass() == H5T_INTEGER) {
            if (ds1Name.empty()) ds1Name = info.name;
            else if (ds2Name.empty()) { ds2Name = info.name; break; }
        }
    }
    
    if (ds1Name.empty() || ds2Name.empty()) {
        std::cout << "Not enough integer datasets for validation" << std::endl;
        return;
    }
    
    // Lire HDF5
    H5::DataSet ds1 = group.openDataSet(ds1Name);
    H5::DataSet ds2 = group.openDataSet(ds2Name);
    
    std::vector<int> h5_d1(nCheck), h5_d2(nCheck);
    
    hsize_t count[1] = {(hsize_t)nCheck};
    hsize_t offset[1] = {0};
    
    H5::DataSpace filespace1 = ds1.getSpace();
    filespace1.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(1, count);
    
    ds1.read(h5_d1.data(), H5::PredType::NATIVE_INT, memspace, filespace1);
    
    H5::DataSpace filespace2 = ds2.getSpace();
    filespace2.selectHyperslab(H5S_SELECT_SET, count, offset);
    ds2.read(h5_d2.data(), H5::PredType::NATIVE_INT, memspace, filespace2);
    
    // Lire ROOT
    TFile* rootf = new TFile(rootfile.c_str());
    TTree* tree = (TTree*)rootf->Get("T2F");
    
    std::string prefix = pathToBranchPrefix(testGroup);
    std::string branch1 = prefix + ds1Name;
    std::string branch2 = prefix + ds2Name;
    
    int root_d1, root_d2;
    tree->SetBranchAddress(branch1.c_str(), &root_d1);
    tree->SetBranchAddress(branch2.c_str(), &root_d2);
    
    // Comparer
    bool allOK = true;
    for (int i = 0; i < nCheck; i++) {
        tree->GetEntry(i);
        
        if (root_d1 != h5_d1[i] || root_d2 != h5_d2[i]) {
            std::cout << "MISMATCH at event " << i << ": "
                      << "HDF5(" << ds1Name << "=" << h5_d1[i] << "," << ds2Name << "=" << h5_d2[i] << ") "
                      << "ROOT(" << ds1Name << "=" << root_d1 << "," << ds2Name << "=" << root_d2 << ")" << std::endl;
            allOK = false;
        }
    }
    
    rootf->Close();
    delete rootf;
    
    if (allOK) {
        std::cout << "✓ Validation PASSED" << std::endl;
    } else {
        std::cout << "✗ Validation FAILED" << std::endl;
    }
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: h5toroot input.h5 output.root [--maxlen 150] [--check]" << std::endl;
        return 1;
    }
    
    std::string h5file = argv[1];
    std::string rootfile = argv[2];
    int maxlen = 150;
    bool doCheck = false;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--maxlen" && i+1 < argc) {
            maxlen = std::stoi(argv[++i]);
        } else if (arg == "--check") {
            doCheck = true;
        }
    }
    
    try {
        convertH5ToRoot(h5file, rootfile, maxlen);
        
        if (doCheck) {
            validateConversion(h5file, rootfile);
        }
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
