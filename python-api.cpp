#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
namespace py = boost::python;
namespace np = boost::python::numpy;

namespace {
    using std::istringstream;
    using std::ostringstream;
    using std::string;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using std::vector;

    // sanity check np::ndarray to make sure they are dense and continuous
    template <typename T=float>
    void check_dense (np::ndarray array, int nd = 0) {
        CHECK(array.get_dtype() == np::dtype::get_builtin<T>());
        if (nd > 0) CHECK(array.get_nd() == nd);
        else nd = array.get_nd();
        int stride = sizeof(T);
        for (int i = 0, off=nd-1; i < nd; ++i, --off) {
            CHECK(array.strides(off) == stride);
            stride *= array.shape(off);
        }
    }

    template <typename T>
    T sqr (T v) {
        return  v * v;
    }

    std::pair<int, int> get_range (float center, float r, int min, int max) {
        min = std::max(int(std::floor(center - r)), min);
        max = std::min(int(std::ceil(center + r)), max);
        return std::make_pair(min, max);
    }

    py::tuple encode (py::tuple shape, py::list nodule) {
        int Z = py::extract<int>(shape[0]);
        int Y = py::extract<int>(shape[1]);
        int X = py::extract<int>(shape[2]);

        float z = py::extract<float>(nodule[0]);
        float y = py::extract<float>(nodule[1]);
        float x = py::extract<float>(nodule[2]);
        float r = py::extract<float>(nodule[3]);


        np::ndarray A = np::zeros(py::make_tuple(Z, Y, X), np::dtype::get_builtin<float>());
        np::ndarray P = np::zeros(py::make_tuple(Z, Y, X, 4), np::dtype::get_builtin<float>());
        check_dense<float>(A);
        check_dense<float>(P);

        auto z_range = get_range(z, r, 0, Z-1);
        for (int i = z_range.first; i <= z_range.second; ++i) {
            float r1 = sqrt(sqrt(r) - sqr(z-i));
            auto y_range = get_range(y, r1, 0, Y-1);
            auto x_range = get_range(x, r1, 0, X-1);

            float *pa = reinterpret_cast<float *>(A.get_data() + i * A.strides(0));
            float *pp = reinterpret_cast<float *>(P.get_data() + i * P.strides(0));
            // slice
            for (int j = y_range.first; j <= y_range.second; ++j) {
                float *ppa = pa + (X * j + x_range.first);
                float *ppp = pp + (X * j + x_range.first)* 4;
                for (int k = x_range.first; k <= x_range.second; ++k) {
                    if (sqr(i - z) + sqr(j - y) + sqr(k - x) <= sqr(r)) {
                        // mask
                        ppa[0] = 1.0;
                        // parameters
                        ppp[0] = z - i;
                        ppp[1] = y - j;
                        ppp[2] = x - k;
                        ppp[3] = r;
                    }
                    ppa += 1;
                    ppp += 4;
                }
            }
        }
        return py::make_tuple(A, P);
    }

    struct Ball {
        float z, y, x, r, s;

        py::tuple to_python () const {
            return py::make_tuple(z, y, x, r, s);
        }
    };

    float overlap (Ball const &a, Ball const &b) {
        // TODO: a better way to calculate this
        float dist = sqrt(sqr(a.x-b.x) + sqr(a.y-b.y) + sqr(a.z-b.z));
        float I = a.r + b.r - dist;
        return I / std::min(a.r, b.r);
    }

    py::list decode (np::ndarray A, np::ndarray P, float anchor_th, float nms_th) {
        check_dense<float>(A);
        check_dense<float>(P);
        float const *pa = reinterpret_cast<float *>(A.get_data());
        float const *pp = reinterpret_cast<float *>(P.get_data());
        vector<Ball> balls;
        for (int i = 0; i < A.shape(0); ++i) {
            for (int j = 0; j < A.shape(1); ++j) {
                for (int k = 0; k < A.shape(2); ++k) {
                    if (pa[0] >= anchor_th) {
                        Ball b;
                        b.z = i + pp[0];
                        b.y = j + pp[1];
                        b.x = k + pp[2];
                        b.r = pp[3];
                        b.s = pa[0];
                        balls.push_back(b);
                    }
                    pa += 1;
                    pp += 4;
                }
            }
        }
        // nms
        vector<Ball> keep;
        std::sort(balls.begin(), balls.end(), [](Ball const &a, Ball const &b) { return a.s > b.s;});
        for (auto const &b: balls) {
            for (auto const &k: keep) {
                if (overlap(b, k) < nms_th) {
                    keep.push_back(b);
                }
            }
        }

        py::list list;
        for (auto const &b: keep) {
            list.append(b.to_python());
        }
        return list;
    }
}

BOOST_PYTHON_MODULE(cpp)
{
    np::initialize();
    def("encode", ::encode);
    def("decode", ::decode);
}

