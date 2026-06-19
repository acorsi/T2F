#include "FragmentPair.h"

FragmentPair::FragmentPair(Fragment&& light, Fragment&& heavy, int eventID)
    : lightFragment_(std::move(light)), heavyFragment_(std::move(heavy)), eventID_(eventID) {
}

std::ostream& operator<<(std::ostream& os, const FragmentPair& fp) {
  os << std::endl  << "FragmentPair(eventID=" << fp.eventID_ << ")\n"
       << "  Light: " << fp.lightFragment_ << "\n"
       << "  Heavy: " << fp.heavyFragment_;
    return os;
}
