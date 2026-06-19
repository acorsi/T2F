#pragma once
#include <vector>

constexpr int MAX_TRACE_LEN = 2560;
constexpr int CHUNK_SIZE    = 10000;
constexpr int MAX_HITS      = 128;    // pads touchés max par fragment

struct PadHit {
    int    row, col;
    double charge;
    double t_start;            // temps du premier franchissement du seuil, ns
    std::vector<double> qt;    // signal Q(t), pas de 10ns, premier point = t_start
};

struct Fragment {
  int    A, Z, n;
  double KE;
  std::vector<double>  raw_trace;
  std::vector<PadHit>  proc_trace;
  double theta, phi;   // direction d'émission (référentiel dérive)
  double pInit[3];     // point d'émission
};

struct Event {
  Fragment L, H;
};
