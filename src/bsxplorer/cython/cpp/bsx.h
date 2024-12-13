#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>

namespace sequence {
    struct ContextData {
        std::vector<long> position;
        std::vector<std::string> trinuc;
        std::vector<std::string> context;
        std::vector<bool> strand;
    };
    ContextData get_trinuc(std::string seq);
    std::vector<std::pair<int, ContextData>> get_trinuc_parallel(std::string seq, int num_threads);
    std::string convert_trinuc(std::string trinuc, bool strand);
}
