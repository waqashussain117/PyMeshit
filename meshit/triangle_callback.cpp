#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm>  // For std::copy
#include <vector>
#include <cmath>
#include <iostream>

// NumPy C API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Triangle callback function signature
typedef double REAL;
typedef REAL* vertex;

// Singleton class to manage mesh parameters and feature points
class GradientControl {
private:
    static GradientControl* instance;
    
    // Parameters
    double gradient;       // Gradient control parameter
    double sq_meshsize;    // Squared base mesh size
    
    // Feature points
    std::vector<double> feature_points_x;
    std::vector<double> feature_points_y;
    std::vector<double> feature_sizes;
    
    // Private constructor
    GradientControl() : gradient(1.0), sq_meshsize(1.0) {}
    
public:
    // Disable copy constructor and assignment operator
    GradientControl(const GradientControl&) = delete;
    GradientControl& operator=(const GradientControl&) = delete;
    
    // Get instance (singleton pattern)
    static GradientControl* getInstance() {
        if (!instance) {
            instance = new GradientControl();
        }
        return instance;
    }
    
    // Initialize with parameters
    void initialize(double g, double ms, 
                   const std::vector<double>& points_x, 
                   const std::vector<double>& points_y, 
                   const std::vector<double>& sizes) {
        gradient = g;
        sq_meshsize = ms; // Already squared meshsize
        
        // Clear and copy feature points
        feature_points_x = points_x;
        feature_points_y = points_y;
        feature_sizes = sizes;
        
        // Adjust feature sizes for more uniform transition
        for (size_t i = 0; i < feature_sizes.size(); i++) {
            // Square the feature sizes for later use
            feature_sizes[i] = feature_sizes[i] * feature_sizes[i];
        }
    }
    
    // Cleanup
    static void cleanup() {
        delete instance;
        instance = nullptr;
    }
    
    // Check if a triangle is suitable
    bool isTriangleSuitable(double x1, double y1, double x2, double y2, double x3, double y3) const {
        // Create edge vectors
        double dxoa = x1 - x3;
        double dyoa = y1 - y3;
        double dxda = x2 - x3;
        double dyda = y2 - y3;
        double dxod = x1 - x2;
        double dyod = y1 - y2;

        // Find squares of edge lengths
        double oalen = dxoa * dxoa + dyoa * dyoa;
        double dalen = dxda * dxda + dyda * dyda;
        double odlen = dxod * dxod + dyod * dyod;
        
        // Find maximum edge length squared
        double max_sq_len = std::max({oalen, dalen, odlen});
        
        // First check against base mesh size
        if (max_sq_len > sq_meshsize) {
            return false;
        }
        
        // Calculate triangle centroid
        constexpr double ONETHIRD = 1.0/3.0;
        double cx = (x1 + x2 + x3) * ONETHIRD;
        double cy = (y1 + y2 + y3) * ONETHIRD;
        
        // Check against feature points with improved sizing function
        if (!feature_points_x.empty()) {
            double sq_grad = gradient * gradient;
            
            // New approach: calculate blended sizing based on multiple feature points
            double blended_target_size = sq_meshsize;  // Start with base size
            double total_weight = 0.0;
            
            for (size_t i = 0; i < feature_points_x.size(); i++) {
                double sq_refinesize = feature_sizes[i];
                double dx = cx - feature_points_x[i];
                double dy = cy - feature_points_y[i];
                double sq_dist = dx*dx + dy*dy;
                
                // Calculate influence radius with smoother transition
                double influence_radius_sq = sq_grad * (sq_meshsize - sq_refinesize);
                
                // Skip if point is outside influence radius
                if (sq_dist >= influence_radius_sq) {
                    continue;
                }
                
                // Calculate normalized distance (0 at feature point, 1 at influence radius)
                double norm_dist = sqrt(sq_dist / influence_radius_sq);
                
                // Use smooth cubic weighting function for transition
                // w(d) = 2d³ - 3d² + 1, which gives a smooth S-curve from 1 to 0
                double weight = 1.0;
                if (norm_dist > 0.0) {
                    weight = 2.0 * norm_dist*norm_dist*norm_dist - 3.0 * norm_dist*norm_dist + 1.0;
                }
                
                // Calculate target size with smoother transition
                double target_size = sq_refinesize + (sq_meshsize - sq_refinesize) * norm_dist;
                
                // Accumulate weighted target size
                blended_target_size = std::min(blended_target_size, target_size);
                total_weight += weight;
            }
            
            // Check if max edge length exceeds the blended target size
            if (max_sq_len > blended_target_size) {
                return false;
            }
        }
        
        return true;
    }
};

// Initialize static instance pointer
GradientControl* GradientControl::instance = nullptr;

// External linkage for Triangle to call our C++ function
extern "C" {
    // This function is called by Triangle to determine if a triangle is suitable
    int triunsuitable(vertex triorg, vertex tridest, vertex triapex, REAL area) {
        GradientControl* gc = GradientControl::getInstance();
        
        // Check if triangle is suitable
        bool is_suitable = gc->isTriangleSuitable(
            triorg[0], triorg[1],
            tridest[0], tridest[1],
            triapex[0], triapex[1]
        );
        
        // Return 1 if unsuitable, 0 if suitable (Triangle's convention)
        return is_suitable ? 0 : 1;
    }
}

// Python wrapper function to initialize the GradientControl
static PyObject* initialize_gradient_control(PyObject* self, PyObject* args, PyObject* kwargs) {
    double gradient;
    double sq_meshsize;
    PyObject* feature_points_obj;
    PyObject* feature_sizes_obj;
    
    static const char* kwlist[] = {"gradient", "sq_meshsize", "feature_points", "feature_sizes", NULL};
    
    // Parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddOO", const_cast<char**>(kwlist),
                                    &gradient, &sq_meshsize, &feature_points_obj, &feature_sizes_obj)) {
        return NULL;
    }
    
    // Convert feature points to C++ vectors
    PyArrayObject* feature_points_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(feature_points_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    
    if (!feature_points_array) {
        return NULL;
    }
    
    PyArrayObject* feature_sizes_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(feature_sizes_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    
    if (!feature_sizes_array) {
        Py_DECREF(feature_points_array);
        return NULL;
    }
    
    // Check dimensions
    if (PyArray_NDIM(feature_points_array) != 2 || PyArray_SHAPE(feature_points_array)[1] != 2) {
        PyErr_SetString(PyExc_ValueError, "Feature points must be an Nx2 array");
        Py_DECREF(feature_points_array);
        Py_DECREF(feature_sizes_array);
        return NULL;
    }
    
    if (PyArray_NDIM(feature_sizes_array) != 1 || 
        PyArray_SHAPE(feature_sizes_array)[0] != PyArray_SHAPE(feature_points_array)[0]) {
        PyErr_SetString(PyExc_ValueError, "Feature sizes must be an array of length N, matching feature points");
        Py_DECREF(feature_points_array);
        Py_DECREF(feature_sizes_array);
        return NULL;
    }
    
    // Extract feature points and sizes
    size_t num_features = PyArray_SHAPE(feature_points_array)[0];
    std::vector<double> points_x(num_features);
    std::vector<double> points_y(num_features);
    std::vector<double> sizes(num_features);
    
    // Copy x-coordinates
    for (size_t i = 0; i < num_features; i++) {
        double* ptr = static_cast<double*>(PyArray_GETPTR2(feature_points_array, i, 0));
        points_x[i] = *ptr;
    }
    
    // Copy y-coordinates
    for (size_t i = 0; i < num_features; i++) {
        double* ptr = static_cast<double*>(PyArray_GETPTR2(feature_points_array, i, 1));
        points_y[i] = *ptr;
    }
    
    // Copy sizes
    for (size_t i = 0; i < num_features; i++) {
        double* ptr = static_cast<double*>(PyArray_GETPTR1(feature_sizes_array, i));
        sizes[i] = *ptr;
    }
    
    // Initialize the GradientControl
    GradientControl::getInstance()->initialize(gradient, sq_meshsize, points_x, points_y, sizes);
    
    // Cleanup
    Py_DECREF(feature_points_array);
    Py_DECREF(feature_sizes_array);
    
    Py_RETURN_NONE;
}

// Module cleanup function
static void cleanup_gradient_control() {
    GradientControl::cleanup();
}

// Method definitions
static PyMethodDef TriangleCallbackMethods[] = {
    {"initialize_gradient_control", (PyCFunction)initialize_gradient_control, 
     METH_VARARGS | METH_KEYWORDS, "Initialize the GradientControl with parameters"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef trianglecallbackmodule = {
    PyModuleDef_HEAD_INIT,
    "triangle_callback",   // Module name
    "C extension for triangle callback",  // Doc string
    -1,                    // Size of per-interpreter state or -1
    TriangleCallbackMethods,
    NULL, NULL, NULL, NULL
};

// Module initialization
PyMODINIT_FUNC PyInit_triangle_callback(void) {
    import_array();  // Initialize NumPy
    
    PyObject* m = PyModule_Create(&trianglecallbackmodule);
    if (m == NULL) {
        return NULL;
    }
    
    // Register cleanup function
    if (PyModule_AddObject(m, "__cleanup", PyCapsule_New((void*)cleanup_gradient_control, 
                                                       "cleanup_gradient_control", NULL)) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
} 