#include <vector>
#include <algorithm>


namespace sequence {
    struct ContextData {
        std::vector<long> position;
        std::vector<std::string> trinuc;
        std::vector<std::string> context;
        std::vector<bool> strand;
    };
    ContextData get_trinuc(std::string seq);
    std::string convert_trinuc(std::string trinuc, bool strand);
}
