#include "Fragment.h"

Fragment::Fragment(int A, int Z, double kineticEnergy, int neutronMultiplicity)
    : A_(A), Z_(Z), kineticEnergy_(kineticEnergy), neutronMultiplicity_(neutronMultiplicity) {
}

std::ostream& operator<<(std::ostream& os, const Fragment& f) {
    os << "Fragment(A=" << f.A_ 
       << ", Z=" << f.Z_ 
       << ", KE=" << f.kineticEnergy_ 
       << ", n=" << f.neutronMultiplicity_ << ")";
    return os;
}

