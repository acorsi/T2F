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
#include <algorithm>
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
TLorentzVector MakeFragmentLV(double A, double Z, double KE, double Eex, double costh, double phi)
{
    constexpr double amu = 931.49410242; // MeV/c2
                                         //double mass = GetMassMeV(A,Z); //A * amu;
    double mass = gef_masses::GetNuclearMassMeV_fromBE(Z, A)+Eex;
    //cout<<Z<<" "<<A<<" "<<mass<<endl;
    double Etot = KE + mass ;
    double p = std::sqrt(KE * (KE + 2.0 * mass));

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
    double p = std::sqrt(T * (T + 2.0 * mn));

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

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << endl << "Usage: exe in_file out_file" << endl;
        exit(1) ;
    }    

    ifstream in(argv[1]);
    if (!in) {
        cerr << "Erreur: impossible d'ouvrir lmd file" << endl;
        return 1;
    }

    ofstream out ; out.open(argv[2],ios::out) ;


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
        float dum, KE1, KE2, KE1pre, KE2pre, TKE, tke, Eex1, Eex2;

        float fission_channel, ZCN, ACN, Eexc;
        float costh1, phi1, costh2, phi2;
        float dummy1, dummy2;
        vn.clear();
        stringstream ev(line);

        ev >> fission_channel >> ZCN >> ACN >> Eexc
            >> Z1 >> Z2 >> a1 >> a2 >> A1 >> A2
            >> dum >> dum >> dum >> dum
            >> KE1pre >> KE1 >> costh1 >> phi1
            >> KE2pre >> KE2 >> costh2 >> phi2
            >> Eex1 >> Eex2
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


        // Rejeter les donnees invalides avant toute racine ou construction de LV.
        auto valid_mass = [](int A, int Z) {
            if (A <= 0 || Z < 0 || Z > A) return false;
            double m = gef_masses::GetNuclearMassMeV_fromBE(Z, A);
            return std::isfinite(m) && m > 0.0;
        };
        auto valid_energy = [](double e) {
            return std::isfinite(e) && e >= 0.0;
        };
        // Seuls les petits depassements d'arrondi sont bornes.
        auto valid_cos = [](double c) {
            return std::isfinite(c) && std::abs(c) <= 1.0 + 1.e-6;
        };
        bool valid_event =
            valid_mass(a1, Z1) && valid_mass(a2, Z2) &&
            valid_mass(A1, Z1) && valid_mass(A2, Z2) &&
            valid_energy(KE1pre) && valid_energy(KE2pre) &&
            valid_energy(KE1) && valid_energy(KE2) &&
            valid_energy(Eex1) && valid_energy(Eex2) &&
            valid_cos(costh1) && valid_cos(costh2) &&
            std::isfinite(phi1) && std::isfinite(phi2);
        for (int i = 0; i < ntot; ++i)
            valid_event = valid_event && valid_energy(vdir[3*i]) &&
                valid_cos(vdir[3*i+1]) && std::isfinite(vdir[3*i+2]);
        if (!valid_event) {
            cerr << "Evenement " << CPT
                 << " ignore : masse, energie ou direction invalide\n";
            ++CPT;
            continue;
        }
        costh1 = std::clamp(costh1, -1.0f, 1.0f);
        costh2 = std::clamp(costh2, -1.0f, 1.0f);

        // Fragment 1 : direction isotrope
        TLorentzVector FF1_pre = MakeFragmentLV(a1, Z1, KE1pre, Eex1, costh1, phi1* M_PI / 180.0);
        TLorentzVector FF1_post_gef = MakeFragmentLV(A1, Z1, KE1, 0,costh1, phi1* M_PI / 180.0);

        // Fragment 2 : direction opposée
        TLorentzVector FF2_pre = MakeFragmentLV(a2, Z2, KE2pre, Eex2, costh2, phi2* M_PI / 180.0);
        TLorentzVector FF2_post_gef = MakeFragmentLV(A2, Z2, KE2, 0,costh2, phi2* M_PI / 180.0);



        out << Z1 << " " << A1 << " " << KE1 << " 0 "
            << Z2 << " " << A2 << " " << KE2 << " 0 "
            << n1 << " " << n2<<" ";

        ////get 4-moment of neutrons and sum them per fragment; in the frame of light fragment then rotated to lab frame //TO ADAPT IF CN FRAME DIFFERENT FROM LAB
        for (int i = 0; i < ntot; i++) {
            //if(i<n1)    cth=vdir[3*i+1];
            //else        cth=-vdir[3*i+1];
            en=vdir[3*i];
            cth=std::clamp(vdir[3*i+1], -1.0f, 1.0f);
            phi=vdir[3*i+2]* M_PI / 180.0;
            //if(i<n1){phif=phi1;costhf=costh1;}
            //else    {phif=phi2;costhf=costh2;}
            auto [theta_lab,phi_lab]=RotateAnglesToLab(acos(cth),phi,acos(costh1),phi1* M_PI / 180.0);
            vn.push_back(MakeNeutronLV(en, cos(theta_lab), phi_lab));
            //cout << "   " << En << " " << cth << " " << phi;
            //if(i<n1)    vn[i].Boost(FF1_pre.BoostVector());
            //else        vn[i].Boost(FF2_pre.BoostVector());
        }

        //build post fragment vectors to get the angle and dump to output file
        TLorentzVector SumN1(0.,0.,0.,0.);
        TLorentzVector SumN2(0.,0.,0.,0.);
        for (int i=0; i<n1; i++)
            SumN1 += vn[i];

        for (int i=n1; i<n1+n2; i++)
            SumN2 += vn[i];

        //Fragments post evaporation
        TLorentzVector FF1_post = FF1_pre-SumN1;
        TLorentzVector FF2_post = FF2_pre-SumN2;
        double angle=(FF1_post.Vect()).Angle(FF2_post.Vect());
        if(std::isnan(angle)) {cerr<<"angle nan"<<endl; angle=M_PI;/* cout<<(FF1_post.Vect()).Mag()<<" "<<(FF2_post.Vect()).Mag()<<endl;break;*/}
        out<<angle<<endl;

        //4 momentum conservation
        TLorentzVector check=FF1_pre+FF2_pre;
        //cerr<<"check "<<check.Mag()-gef_masses::GetNuclearMassMeV_fromBE(98,252)<<endl;
        //if(n1==0|| n2==0)cerr<<"Fragments :"<<FF1_post.Mag()-FF1_post_gef.Mag()<<" "<<Eex1<<"; "<<FF2_post.Mag()-FF2_post_gef.Mag()<<" "<<Eex2<<endl;
        //cout<<"check "<<check.Px()<<" "<<check.Py()<<" "<<check.Pz()<<" "<<check.M()<<" "<<Eexc<<" "<<gef_masses::GetNuclearMassMeV_fromBE(98,252)-gef_masses::GetNuclearMassMeV_fromBE(Z1,a1)-gef_masses::GetNuclearMassMeV_fromBE(Z2,a2)<<" "<<tke<<endl;
        //
        double KE1_reco = FF1_post.E() - FF1_post.M();
        double KE2_reco = FF2_post.E() - FF2_post.M();
        if( abs(KE1_reco - KE1)>5||abs(KE2_reco - KE2)>5) cerr << "Delta KE = " << KE1_reco - KE1 <<" or "<<KE2_reco - KE2 << " MeV\n";
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
