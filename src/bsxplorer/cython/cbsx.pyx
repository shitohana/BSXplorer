# distutils: language = c++
# distutils: sources = src/bsxplorer/cython/cpp/sequence.cpp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
import itertools

cdef extern from "bsx.h" namespace "sequence":
    cdef string convert_trinuc(string trinuc, bool strand)
    cdef cppclass ContextData:
        vector[long] position
        vector[bool] strand
        vector[string]  context
        vector[string]  trinuc

    cdef vector[pair[int, ContextData]] get_trinuc_parallel(string seq, int num_threads)
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

def get_trinuc_parallel_cython(str record_seq, int num_threads):
    record_seq_b = record_seq.encode('ascii')
    cdef vector[pair[int, ContextData]] results = get_trinuc_parallel(string(<char*> record_seq_b), num_threads)
    cdef list results_python = []
    for res in results:
        results_python.append((res.first, (res.second.position, res.second.strand, res.second.context, res.second.trinuc)))
    results_python.sort(key=lambda r: r[0])
    return (
        list(itertools.chain(*[res[1][0] for res in results_python])),
        list(itertools.chain(*[res[1][1] for res in results_python])),
        list(itertools.chain(*list(map(lambda s: s.decode('ascii'), res[1][2]) for res in results_python))),
        list(itertools.chain(*list(map(lambda s: s.decode('ascii'), res[1][3]) for res in results_python))),
    )
