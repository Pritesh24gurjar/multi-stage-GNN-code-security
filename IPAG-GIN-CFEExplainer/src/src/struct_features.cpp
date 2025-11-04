#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace py = pybind11;

struct StructFeatures {
    int degree;
    int in_degree;
    int out_degree;
    bool is_leaf;
    bool is_root;
};

std::unordered_map<std::string, StructFeatures> compute_structural_features_bulk(
    const std::vector<std::string>& node_ids,
    const std::vector<std::pair<std::string, std::string>>& edges
) {
    std::unordered_map<std::string, int> in_counts;
    std::unordered_map<std::string, int> out_counts;

    // Count in-degrees and out-degrees
    for (const auto& e : edges) {
        out_counts[e.first]++;
        in_counts[e.second]++;
    }

    // Build feature map for all nodes
    std::unordered_map<std::string, StructFeatures> features;
    for (const auto& node_id : node_ids) {
        int in_degree = in_counts[node_id];
        int out_degree = out_counts[node_id];
        int degree = in_degree + out_degree;
        features[node_id] = StructFeatures{
            degree, 
            in_degree, 
            out_degree, 
            out_degree == 0,  // is_leaf
            in_degree == 0    // is_root
        };
    }
    return features;
}

PYBIND11_MODULE(graph_structure_features, m) {
    m.doc() = "C++ accelerated structural feature computation for IPAG graphs";
    
    py::class_<StructFeatures>(m, "StructFeatures")
        .def_readonly("degree", &StructFeatures::degree)
        .def_readonly("in_degree", &StructFeatures::in_degree)
        .def_readonly("out_degree", &StructFeatures::out_degree)
        .def_readonly("is_leaf", &StructFeatures::is_leaf)
        .def_readonly("is_root", &StructFeatures::is_root);

    m.def("compute_structural_features_bulk", 
          &compute_structural_features_bulk, 
          py::arg("node_ids"),
          py::arg("edges"),
          "Bulk compute structural features for graph nodes");
}