#include "bsx.h"
#include <omp.h>

std::string sequence::convert_trinuc(std::string trinuc, bool strand) {
    std::transform(trinuc.begin(), trinuc.end(), trinuc.begin(), ::toupper);
    
    if (strand) {
        return trinuc;
    } else {
        std::reverse(trinuc.begin(), trinuc.end());
        std::string out = "C";
        char converted;
        for (int i = 1; i < trinuc.size(); ++i) {
            switch (trinuc[i]) {
                case 'C': out += 'G'; break;
                case 'G': out += 'C'; break;
                case 'A': out += 'T'; break;
                case 'T': out += 'A'; break;
                case 'U': out += 'R'; break;
                case 'R': out += 'U'; break;
                case 'Y': out += 'K'; break;
                case 'K': out += 'Y'; break;
                case 'S': out += 'S'; break;
                case 'W': out += 'W'; break;
                default: out += 'N';  break;
            };
        };
        return out;
    }
}

sequence::ContextData sequence::get_trinuc(std::string seq) {
    ContextData data;
    long rv_bound = 1;
    long fw_bound = seq.size() - 3;
    std::string trinuc;
    char c;
    bool strand;

    for (int i = 0; i < seq.size(); ++i) {
        c = std::toupper(seq[i]);
        if (!(c == 'C' || c == 'G')) continue;
        
        strand = (c == 'C');
        try {
            trinuc = convert_trinuc(seq.substr(strand ? i : i - 2, 3), strand);
        } catch (const std::out_of_range& e) { continue; };
        

        data.position.push_back(i + 1);
        data.strand.push_back(strand);
        data.trinuc.push_back(trinuc);
        if      (trinuc[1] == 'G') { data.context.push_back("CG");  }
        else if (trinuc[2] == 'G') { data.context.push_back("CHG"); }
        else                       { data.context.push_back("CHH"); }
    };
    return data;
};