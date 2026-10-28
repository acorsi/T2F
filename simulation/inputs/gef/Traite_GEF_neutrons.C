#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "TLorentzVector.h"
#include "TRandom3.h"
#include "TH2F.h"
#include "TFile.h"
#include "BEexp_constexpr.hpp"
#include <utility>
using namespace std;

#define NMAX 50000000

// /usr/bin/clang++ -std=c++17 -O2 -Wall $(root-config --cflags) -o Traite_GEF_neutrons Traite_GEF_neutrons.C $(root-config --libs)

std::pair<double,double> RotateAnglesToLab(double theta,
                                           double phi,
                                           double Theta,
                                           double Phi)
{
    // vecteur unitaire dans le repère local
    double st = std::sin(theta);
    double xp = st * std::cos(phi);
    double yp = st * std::sin(phi);
    double zp = std::cos(theta);

    // rotation du repère local vers labo: Rz(Phi) Ry(Theta)
    double cT = std::cos(Theta);
    double sT = std::sin(Theta);
    double cP = std::cos(Phi);
    double sP = std::sin(Phi);

    double x =
        cP*cT*xp - sP*yp + cP*sT*zp;

    double y =
        sP*cT*xp + cP*yp + sP*sT*zp;

    double z =
       -sT*xp + cT*zp;

    // sécurité numérique
    if (z >  1.0) z =  1.0;
    if (z < -1.0) z = -1.0;

    double theta_lab = std::acos(z);
    double phi_lab   = std::atan2(y, x);

    // optionnel: ramener phi dans [0,2pi[
    if (phi_lab < 0)
        phi_lab += 2.0*M_PI;

    return {theta_lab, phi_lab};
}

bool is_blank_or_comment(const string& s) {
    size_t p = s.find_first_not_of(" \t\r\n");
    if (p == string::npos) return true;
    return s[p] == '*';
}

vector<float> read_particle_line(const string& line, int& tag) {
    vector<float> values;
    stringstream ss(line);

    ss >> tag;

    float x;
    while (ss >> x) values.push_back(x);

    return values;
}
TLorentzVector MakeFragmentLV(double A, double Z, double KE, double costh, double phi)
{
    constexpr double amu = 931.49410242; // MeV/c2
    //double mass = GetMassMeV(A,Z); //A * amu;
    double mass = gef_masses::GetNuclearMassMeV_fromBE(Z, A);
    //cout<<Z<<" "<<A<<" "<<mass<<endl;
    double Etot = KE + mass;
    double p = std::sqrt(Etot * Etot - mass * mass);

    double sinth = std::sqrt(std::max(0.0, 1.0 - costh * costh));

    double px = p * sinth * std::cos(phi);
    double py = p * sinth * std::sin(phi);
    double pz = p * costh;

    return TLorentzVector(px, py, pz, Etot);
}
TLorentzVector MakeNeutronLV(double T, double costh, double phi)
{
    constexpr double mn = 939.56542052; // MeV/c^2

    double Etot = T + mn;
    double p = std::sqrt(Etot * Etot - mn * mn);

    double sinth2 = 1.0 - costh * costh;
    if (sinth2 < 0.0) sinth2 = 0.0; // protection numérique
    double sinth = std::sqrt(sinth2);

    double px = p * sinth * std::cos(phi);
    double py = p * sinth * std::sin(phi);
    double pz = p * costh;

    TLorentzVector v;
    v.SetPxPyPzE(px, py, pz, Etot);
    return v;
}
int main() {

    ifstream in("y.lmd");
    if (!in) {
        cerr << "Erreur: impossible d'ouvrir test.lmd" << endl;
        return 1;
    }

    string line;
    int CPT = 0;
    float cth, phi, en;
    TLorentzVector vf1, vf2;
    vector<TLorentzVector> vn;
    TH1F*h=new TH1F("h","Angle pre - Angle post",1000,-5,5);
    TH2F*h2=new TH2F("h2","Angle pre - Angle post vs neutrons",10,0,10,1000,-5,5);
    while (getline(in, line)) {

        if (is_blank_or_comment(line)) continue;

        int A1, Z1, A2, Z2, a1, a2, n1, n2;
        float dum, KE1, KE2, TKE, tke;

        float fission_channel, ZCN, ACN, Eexc;
        float costh1, phi1, costh2, phi2;
        float dummy1, dummy2;
        vn.clear();
        stringstream ev(line);

        ev >> fission_channel >> ZCN >> ACN >> Eexc
            >> Z1 >> Z2 >> a1 >> a2 >> A1 >> A2
            >> dum >> dum >> dum >> dum
            >> dum >> KE1 >> costh1 >> phi1
            >> dum >> KE2 >> costh2 >> phi2
            >> dummy1 >> dummy2
            >> n1 >> n2 >> tke >> TKE >> dum;

        if (!ev) {
            cerr << "Erreur lecture evenement " << CPT << endl;
            cerr << "Ligne fautive = [" << line << "]" << endl;
            return 1;
        }

        string ldir, lfrag1, lfrag2;

        if (!getline(in, ldir) || !getline(in, lfrag1) || !getline(in, lfrag2)) {
            cerr << "Erreur: lignes neutron manquantes apres evenement " << CPT << endl;
            return 1;
        }

        int tag_dir, tag_f1, tag_f2;

        vector<float> vdir = read_particle_line(ldir, tag_dir);//Energies [MeV] in CN frame and angles vs. direction of light fragment for all post-scission neutrons
        vector<float> v1   = read_particle_line(lfrag1, tag_f1);//Energies [MeV] of neutrons emitted from light fragment in frame of light fragment and in frame of CN.
        vector<float> v2   = read_particle_line(lfrag2, tag_f2);//same for heavy fragment

        int ntot = n1 + n2;

        if ((int)vdir.size() < 3 * ntot) {
            cerr << "Erreur: ligne directions trop courte, event " << CPT
                << " n1=" << n1 << " n2=" << n2
                << " vdir.size()=" << vdir.size()
                << " attendu=" << 3 * ntot << endl;
            return 1;
        }

        if ((int)v1.size() < 2 * n1) {
            cerr << "Erreur: ligne fragment 1 trop courte, event " << CPT
                << " n1=" << n1
                << " v1.size()=" << v1.size()
                << " attendu=" << 2 * n1 << endl;
            return 1;
        }

        if ((int)v2.size() < 2 * n2) {
            cerr << "Erreur: ligne fragment 2 trop courte, event " << CPT
                << " n2=" << n2
                << " v2.size()=" << v2.size()
                << " attendu=" << 2 * n2 << endl;
            return 1;
        }

        cout << Z1 << " " << A1 << " " << KE1 << " 0 "
            << Z2 << " " << A2 << " " << KE2 << " 0 "
            << n1 << " " << n2<<" ";

        // Fragment 1 : direction isotrope
        TLorentzVector FF1_pre = MakeFragmentLV(a1, Z1, KE1, costh1, phi1* M_PI / 180.0);

        // Fragment 2 : direction opposée
        TLorentzVector FF2_pre = MakeFragmentLV(a2, Z2, KE2, costh2, phi2* M_PI / 180.0);

        ////get 4-moment of neutrons and sum them per fragment
        vector<float> En;
        for (int i = 0; i < n1; i++) {
            En.push_back(v1[2*i]);
        }
        for (int i = 0; i < n2; i++) {
            En.push_back(v2[2*i]);
        }
        double phif, costhf;
        
        for (int i = 0; i < ntot; i++) {
            //if(i<n1)    cth=vdir[3*i+1];
            //else        cth=-vdir[3*i+1];
            en=vdir[3*i];
            cth=vdir[3*i+1];
            phi=vdir[3*i+2]* M_PI / 180.0;
            //if(i<n1){phif=phi1;costhf=costh1;}
            //else    {phif=phi2;costhf=costh2;}
            auto [theta_lab,phi_lab]=RotateAnglesToLab(acos(cth),phi,acos(costh1),phi1* M_PI / 180.0);
            vn.push_back(MakeNeutronLV(en, cos(theta_lab), phi_lab));
            //cout << "   " << En << " " << cth << " " << phi;
            if(i<n1)    vn[i].Boost(FF1_pre.BoostVector());
            else        vn[i].Boost(FF2_pre.BoostVector());
        }
       /* 
        for (int i = 0; i < ntot; i++) {
            //En=vdir[3*i];
            if(i<n1)    cth=vdir[3*i+1];
            else        cth=-vdir[3*i+1];
            phi=vdir[3*i+2]* M_PI / 180.0;
            vn.push_back(MakeNeutronLV(En.at(i), cth, phi));
        }
        */
        //cout<<"****"<<FF2_pre.BoostVector().Px()<<" "<<FF2_pre.BoostVector().Py()<<" "<<FF2_pre.BoostVector().Pz()<<endl;
        TLorentzVector SumN1(0.,0.,0.,0.);
        TLorentzVector SumN2(0.,0.,0.,0.);
        for (int i=0; i<n1; i++)
            SumN1 += vn[i];

        for (int i=n1; i<n1+n2; i++)
            SumN2 += vn[i];


        

        //Fragments post evaporation
        TLorentzVector FF1_post = FF1_pre-SumN1;
        TLorentzVector FF2_post = FF2_pre-SumN2;
        /*cout<<"("; 
        cout<<FF1_pre.Px()<<" "<<FF1_pre.Py()<<" "<<FF1_pre.Pz()<<" "<<FF1_pre.Mag()<<" ";
        cout<<")"<<" (";
        cout<<FF1_post.Px()<<" "<<FF1_post.Py()<<" "<<FF1_post.Pz()<<" "<<FF1_post.Mag()<<" ";
        cout<<")"<<" (";
        cout<<FF2_pre.Px()<<" "<<FF2_pre.Py()<<" "<<FF2_pre.Pz()<<" ";
        cout<<")"<<" (";
        cout<<FF2_post.Px()<<" "<<FF2_post.Py()<<" "<<FF2_post.Pz()<<" ";
        cout<<")";
        cout << endl;
        cout<<"check "<<check.Px()<<" "<<check.Py()<<" "<<check.Pz()<<" "<<check.M()<<" "<<Eexc<<" "<<gef_masses::GetNuclearMassMeV_fromBE(98,252)-gef_masses::GetNuclearMassMeV_fromBE(Z1,a1)-gef_masses::GetNuclearMassMeV_fromBE(Z2,a2)<<" "<<tke<<endl;
        */
        //cout<<"costheta: "<<(FF1_pre.Vect()).Angle(FF2_pre.Vect())<<" "<<(FF1_post.Vect()).Angle(FF2_post.Vect())<<endl;
        double angle=(FF1_post.Vect()).Angle(FF2_post.Vect());
        if(std::isnan(angle)) {angle=M_PI;/* cout<<(FF1_post.Vect()).Mag()<<" "<<(FF2_post.Vect()).Mag()<<endl;break;*/}
        cout<<angle<<endl;
        
        h->Fill(((FF1_pre.Vect()).Angle(FF2_pre.Vect())-(FF1_post.Vect()).Angle(FF2_post.Vect()))/ M_PI * 180.0);
        h2->Fill(n1+n2,((FF1_pre.Vect()).Angle(FF2_pre.Vect())-(FF1_post.Vect()).Angle(FF2_post.Vect()))/ M_PI * 180.0);

        if (n1 != a1 - A1) {
            cerr << "Attention event " << CPT << ": n1 != a1-A1 : "
                << n1 << " " << a1 << " " << A1 << endl;
        }

        if (n2 != a2 - A2) {
            cerr << "Attention event " << CPT << ": n2 != a2-A2 : "
                << n2 << " " << a2 << " " << A2 << endl;
        }

        if (a1 + a2 != 252) {
            cerr << "Attention event " << CPT << ": a1+a2 != 252 : "
                << a1 << " " << a2 << endl;
        }

        CPT++;
        if (CPT > NMAX) break;
    }


    in.close();
    TFile f("fout.root", "RECREATE");
    h->Write();
    h2->Write();
    return 0;
}
