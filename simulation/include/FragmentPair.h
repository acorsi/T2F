#ifndef FRAGMENT_PAIR_H
#define FRAGMENT_PAIR_H

#include "Fragment.h"

class FragmentPair {
private:
    Fragment lightFragment_;
    Fragment heavyFragment_;
    int eventID_;

public:
  // Constructeur avec move semantics
  FragmentPair(Fragment&& light, Fragment&& heavy, int eventID);

  // Redefinition << pour Print
  friend std::ostream& operator<<(std::ostream& os, const FragmentPair& f);

  // Suppression explicite du copy constructor
  FragmentPair(const FragmentPair&) = delete;
  FragmentPair& operator=(const FragmentPair&) = delete;
  
  // Move constructor et assignment (par défaut OK)
  FragmentPair(FragmentPair&&) = default;
  FragmentPair& operator=(FragmentPair&&) = default;
  
  // Getters
  Fragment& getLightFragment() { return lightFragment_; }
  const Fragment& getLightFragment() const { return lightFragment_; }
  
  Fragment& getHeavyFragment() { return heavyFragment_; }
  const Fragment& getHeavyFragment() const { return heavyFragment_; }
  
  int getEventID() const { return eventID_; }
  
  void printFragmentPair() ;
};

#endif // FRAGMENT_PAIR_H
