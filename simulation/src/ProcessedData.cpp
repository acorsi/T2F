#include "ProcessedData.h"
#include <stdexcept>

ProcessedData::ProcessedData(Type type) : type_(type) {
}
/*
ProcessedData ProcessedData::fromFeatures(const FeatureSet& features) {
    ProcessedData data(Type::FEATURES);
    data.features_ = features;
    return data;
}
*//
ProcessedData ProcessedData::fromLatent(const std::vector<double>& latent) {
    ProcessedData data(Type::LATENT);
    data.latentVector_ = latent;
    return data;
}
/*
const FeatureSet& ProcessedData::getFeatures() const {
    if (type_ != Type::FEATURES) {
        throw std::runtime_error("ProcessedData is not FEATURES type");
    }
    return features_;
}
*/
const std::vector<double>& ProcessedData::getLatent() const {
    if (type_ != Type::LATENT) {
        throw std::runtime_error("ProcessedData is not LATENT type");
    }
    return latentVector_;
}
