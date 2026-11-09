#pragma once

// ---------------------------------------------------------------------------
// Lecteur de fichier d'input : lignes "cle = valeur", commentaires avec '#'.
//
//   - une cle inconnue (faute de frappe) -> erreur via check_unused()
//   - une valeur non numerique           -> erreur
//   - une cle en double                  -> erreur
//   - set_override("cle=valeur") permet de surcharger depuis la ligne de commande
//
// C++ classique : pas de dependance, tout dans le header.
// ---------------------------------------------------------------------------

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

class ConfigFile {
public:
    // Lecture du fichier. Garde aussi le texte brut (pour les metadonnees de sortie).
    void load(const std::string& path) {
        std::ifstream in(path.c_str());
        if (!in) throw std::runtime_error("ConfigFile : impossible d'ouvrir " + path);
        std::string line;
        int n = 0;
        while (std::getline(in, line)) {
            ++n;
            m_text += line + "\n";
            std::string s = strip_comment(line);
            s = trim(s);
            if (s.empty()) continue;
            std::ostringstream where;
            where << path << " ligne " << n;
            parse_assignment(s, where.str());
        }
    }

    // "cle=valeur" (ligne de commande) : remplace la valeur lue dans le fichier
    void set_override(const std::string& assignment) {
        std::string s = trim(assignment);
        size_t eq = s.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("ConfigFile : argument '" + assignment + "' : attendu cle=valeur");
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        Entry e;
        e.value = val;
        e.used = false;
        m_map[key] = e;
        m_text += "# surcharge ligne de commande\n" + key + " = " + val + "\n";
    }

    bool has(const std::string& key) const {
        return m_map.find(key) != m_map.end();
    }

    // --- cle obligatoire ---
    std::string get_string(const std::string& key) {
        std::map<std::string, Entry>::iterator it = m_map.find(key);
        if (it == m_map.end())
            throw std::runtime_error("ConfigFile : cle obligatoire absente : " + key);
        it->second.used = true;
        return it->second.value;
    }

    // --- cle optionnelle avec valeur par defaut ---
    std::string get_string(const std::string& key, const std::string& def) {
        if (!has(key)) return def;
        return get_string(key);
    }

    double get_double(const std::string& key, double def) {
        if (!has(key)) return def;
        std::string s = get_string(key);
        char* end = 0;
        double v = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0')
            throw std::runtime_error("ConfigFile : '" + key + "' : '" + s + "' n'est pas un nombre");
        return v;
    }

    int get_int(const std::string& key, int def) {
        if (!has(key)) return def;
        std::string s = get_string(key);
        char* end = 0;
        long v = std::strtol(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0')
            throw std::runtime_error("ConfigFile : '" + key + "' : '" + s + "' n'est pas un entier");
        return (int)v;
    }

    // A appeler apres toutes les lectures : leve une erreur si une cle n'a jamais ete lue
    // (typiquement une faute de frappe, qui serait sinon ignoree sans message).
    void check_unused() const {
        std::string bad;
        for (std::map<std::string, Entry>::const_iterator it = m_map.begin();
             it != m_map.end(); ++it)
            if (!it->second.used) bad += " " + it->first;
        if (!bad.empty())
            throw std::runtime_error("ConfigFile : cle(s) inconnue(s) :" + bad);
    }

    // Texte complet (fichier + surcharges), pour les metadonnees
    const std::string& text() const { return m_text; }

private:
    struct Entry {
        std::string value;
        bool used;
    };
    std::map<std::string, Entry> m_map;
    std::string m_text;

    static std::string trim(const std::string& s) {
        const char* ws = " \t\r\n";
        size_t a = s.find_first_not_of(ws);
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(ws);
        return s.substr(a, b - a + 1);
    }

    static std::string strip_comment(const std::string& s) {
        size_t h = s.find('#');
        return (h == std::string::npos) ? s : s.substr(0, h);
    }

    void parse_assignment(const std::string& s, const std::string& where) {
        size_t eq = s.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("ConfigFile : " + where + " : attendu cle = valeur");
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        if (key.empty())
            throw std::runtime_error("ConfigFile : " + where + " : cle vide");
        if (m_map.find(key) != m_map.end())
            throw std::runtime_error("ConfigFile : " + where + " : cle en double : " + key);
        Entry e;
        e.value = val;
        e.used = false;
        m_map[key] = e;
    }
};
