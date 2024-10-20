# distutils: language = c++
# distutils: sources = src/bsxplorer/cython/cpp/sequence.cpp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "bsx.h" namespace "sequence":
    cdef string convert_trinuc(string trinuc, bool strand)
    cdef cppclass ContextData:
        vector[long] position
        vector[bool] strand
        vector[string]  context
        vector[string]  trinuc

    cdef ContextData get_trinuc(string seq)


def convert_trinuc_cython(str trinuc, bint strand):
    trinuc_b = trinuc.encode('ascii')
    return convert_trinuc(string(<char*> trinuc_b), strand)

def get_trinuc_cython(str record_seq):
    record_seq_b = record_seq.encode('ascii')
    result = get_trinuc(string(<char*> record_seq_b))
    return (
        result.position, 
        result.strand, 
        list(map(lambda s: s.decode('ascii'), result.context)), 
        list(map(lambda s: s.decode('ascii'), result.trinuc))
        )