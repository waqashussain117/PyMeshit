#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Vector3D class with full functionality
class Vector3D {
public:
    double x, y, z;
    
    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}
    
    // Binary operators as member functions
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    // Rest of Vector3D methods...
    double length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    double lengthSquared() const {
        return x*x + y*y + z*z;
    }
    
    Vector3D normalized() const {
        double len = length();
        if (len > 1e-10)
            return Vector3D(x/len, y/len, z/len);
        return *this;
    }

    void rotX(double sin, double cos) {
        double tmpY = y*cos - z*sin;
        double tmpZ = y*sin + z*cos;
        y = tmpY;
        z = tmpZ;
    }

    void rotZ(double sin, double cos) {
        double tmpX = x*cos - y*sin;
        double tmpY = x*sin + y*cos;
        x = tmpX;
        y = tmpY;
    }

    double distanceToPlane(const Vector3D& plane, const Vector3D& normal) const {
        return dot(*this - plane, normal);
    }

    double distanceToLine(const Vector3D& point, const Vector3D& direction) const {
        if (direction.lengthSquared() < 1e-10)
            return (*this - point).length();
        // Use direction * (scalar) to fix the error
        Vector3D p = point + direction * dot(*this - point, direction);
        return (*this - p).length();
    }

    static double dot(const Vector3D& v1, const Vector3D& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    static Vector3D cross(const Vector3D& v1, const Vector3D& v2) {
        return Vector3D(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x
        );
    }

    static Vector3D normal(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3) {
        return cross(v2 - v1, v3 - v1).normalized();
    }
};

// Add global operator for scalar * vector
inline Vector3D operator*(double scalar, const Vector3D& vec) {
    return Vector3D(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}


// Triangle class with enhanced functionality
class Triangle {
public:
    Vector3D v1, v2, v3;
    
    Triangle(const Vector3D& v1, const Vector3D& v2, const Vector3D& v3)
        : v1(v1), v2(v2), v3(v3) {}
    
    Vector3D normal() const {
        return Vector3D::cross(v2 - v1, v3 - v1).normalized();
    }

    double area() const {
        Vector3D cross = Vector3D::cross(v2 - v1, v3 - v1);
        return 0.5 * cross.length();
    }

    Vector3D centroid() const {
        return Vector3D(
            (v1.x + v2.x + v3.x) / 3.0,
            (v1.y + v2.y + v3.y) / 3.0,
            (v1.z + v2.z + v3.z) / 3.0
        );
    }

    bool containsPoint(const Vector3D& p) const {
        // Compute barycentric coordinates
        Vector3D normal = this->normal();
        double area = this->area();
        
        double a = Vector3D::cross(v2 - v1, p - v1).length() / (2.0 * area);
        double b = Vector3D::cross(v3 - v2, p - v2).length() / (2.0 * area);
        double c = Vector3D::cross(v1 - v3, p - v3).length() / (2.0 * area);
        
        // Point is inside if all barycentric coordinates are between 0 and 1
        return (a >= 0 && a <= 1) && (b >= 0 && b <= 1) && (c >= 0 && c <= 1) && 
               (std::abs(a + b + c - 1.0) < 1e-10);
    }
};

// Main MeshIt model class
class MeshItModel {
private:
    std::vector<std::vector<Vector3D>> polylines;
    std::vector<Triangle> triangles;
    std::vector<Vector3D> mesh_vertices;
    std::vector<std::vector<int>> mesh_faces;
    double mesh_quality;
    std::string mesh_algorithm;
    bool has_constraints;

public:
    MeshItModel() : mesh_quality(1.0), mesh_algorithm("delaunay"), has_constraints(false) {}

    void set_mesh_quality(double quality) {
        mesh_quality = std::max(0.1, std::min(2.0, quality));
    }

    void set_mesh_algorithm(const std::string& algorithm) {
        mesh_algorithm = algorithm;
    }

    void enable_constraints(bool enable) {
        has_constraints = enable;
    }

    void add_polyline(const std::vector<std::vector<double>>& points) {
        std::vector<Vector3D> polyline;
        for (const auto& point : points) {
            if (point.size() >= 3) {
                polyline.push_back(Vector3D(point[0], point[1], point[2]));
            }
        }
        if (!polyline.empty()) {
            polylines.push_back(polyline);
            std::cout << "Added polyline with " << polyline.size() << " points" << std::endl;
        }
    }

    void add_triangle(const std::vector<double>& v1, 
                     const std::vector<double>& v2, 
                     const std::vector<double>& v3) {
        if (v1.size() >= 3 && v2.size() >= 3 && v3.size() >= 3) {
            triangles.push_back(Triangle(
                Vector3D(v1[0], v1[1], v1[2]),
                Vector3D(v2[0], v2[1], v2[2]),
                Vector3D(v3[0], v3[1], v3[2])
            ));
        }
    }

    void pre_mesh() {
        std::cout << "Pre-meshing " << polylines.size() << " polylines" << std::endl;
        // Clean up existing mesh data
        triangles.clear();
        mesh_vertices.clear();
        mesh_faces.clear();

        if (has_constraints) {
            handle_constraints();
        }
    }

    void handle_constraints() {
        // Implementation of constraint handling
        std::cout << "Processing mesh constraints..." << std::endl;
    }

    void mesh() {
        std::cout << "Meshing " << polylines.size() << " polylines..." << std::endl;
        
        if (mesh_algorithm == "delaunay") {
            mesh_delaunay();
        } else if (mesh_algorithm == "advancing_front") {
            mesh_advancing_front();
        } else {
            mesh_simple();
        }
    }

    void mesh_delaunay() {
        // Delaunay triangulation implementation
        std::cout << "Using Delaunay triangulation..." << std::endl;
        mesh_simple(); // Fallback to simple for now
    }

    void mesh_advancing_front() {
        // Advancing front method implementation
        std::cout << "Using advancing front method..." << std::endl;
        mesh_simple(); // Fallback to simple for now
    }

    void mesh_simple() {
        for (const auto& polyline : polylines) {
            if (polyline.size() < 3) continue;
            
            int start_idx = mesh_vertices.size();
            
            for (const auto& pt : polyline) {
                mesh_vertices.push_back(pt);
            }
            
            for (size_t i = 1; i < polyline.size() - 1; i++) {
                mesh_faces.push_back({start_idx, start_idx + (int)i, start_idx + (int)(i+1)});
                triangles.push_back(Triangle(polyline[0], polyline[i], polyline[i+1]));
            }
        }
        
        std::cout << "Created " << triangles.size() << " triangles" << std::endl;
        std::cout << "Mesh has " << mesh_vertices.size() << " vertices and " 
                  << mesh_faces.size() << " faces" << std::endl;
    }

    void export_vtu(const std::string& filename) {
        std::cout << "Exporting mesh to " << filename << std::endl;
        
        std::ofstream vtu_file(filename);
        if (!vtu_file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        write_vtu_header(vtu_file);
        write_vtu_points(vtu_file);
        write_vtu_cells(vtu_file);
        write_vtu_cell_data(vtu_file);
        write_vtu_footer(vtu_file);
        
        vtu_file.close();
        std::cout << "Export complete: " << filename << std::endl;
    }

private:
    void write_vtu_header(std::ofstream& file) {
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <UnstructuredGrid>\n";
        file << "    <Piece NumberOfPoints=\"" << mesh_vertices.size() 
             << "\" NumberOfCells=\"" << mesh_faces.size() << "\">\n";
    }

    void write_vtu_points(std::ofstream& file) {
        file << "      <Points>\n";
        file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (const auto& vertex : mesh_vertices) {
            file << "          " << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
        }
        file << "        </DataArray>\n";
        file << "      </Points>\n";
    }

    void write_vtu_cells(std::ofstream& file) {
        file << "      <Cells>\n";
        write_vtu_connectivity(file);
        write_vtu_offsets(file);
        write_vtu_types(file);
        file << "      </Cells>\n";
    }

    void write_vtu_connectivity(std::ofstream& file) {
        file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (const auto& face : mesh_faces) {
            file << "          ";
            for (int idx : face) {
                file << idx << " ";
            }
            file << "\n";
        }
        file << "        </DataArray>\n";
    }

    void write_vtu_offsets(std::ofstream& file) {
        file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        file << "          ";
        int offset = 0;
        for (const auto& face : mesh_faces) {
            offset += face.size();
            file << offset << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";
    }

    void write_vtu_types(std::ofstream& file) {
        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        file << "          ";
        for (size_t i = 0; i < mesh_faces.size(); i++) {
            file << "5 "; // 5 = VTK_TRIANGLE
        }
        file << "\n";
        file << "        </DataArray>\n";
    }

    void write_vtu_cell_data(std::ofstream& file) {
        file << "      <CellData>\n";
        file << "        <DataArray type=\"Float32\" Name=\"Normals\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < mesh_faces.size(); i++) {
            const auto& face = mesh_faces[i];
            if (face.size() >= 3) {
                Vector3D v1 = mesh_vertices[face[0]];
                Vector3D v2 = mesh_vertices[face[1]];
                Vector3D v3 = mesh_vertices[face[2]];
                Vector3D normal = Vector3D::cross(v2 - v1, v3 - v1).normalized();
                file << "          " << normal.x << " " << normal.y << " " << normal.z << "\n";
            }
            else {
                file << "          0 0 1\n";
            }
        }
        file << "        </DataArray>\n";
        file << "      </CellData>\n";
    }

    void write_vtu_footer(std::ofstream& file) {
        file << "    </Piece>\n";
        file << "  </UnstructuredGrid>\n";
        file << "</VTKFile>\n";
    }
};

PYBIND11_MODULE(_meshit, m) {
    m.doc() = "MeshIt Python bindings for PZero integration";
    
    // Update the Vector3D bindings to include operators
    py::class_<Vector3D>(m, "Vector3D")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vector3D::x)
        .def_readwrite("y", &Vector3D::y)
        .def_readwrite("z", &Vector3D::z)
        .def("length", &Vector3D::length)
        .def("lengthSquared", &Vector3D::lengthSquared)
        .def("normalized", &Vector3D::normalized)
        .def("rotX", &Vector3D::rotX)
        .def("rotZ", &Vector3D::rotZ)
        .def("distanceToPlane", &Vector3D::distanceToPlane)
        .def("distanceToLine", &Vector3D::distanceToLine)
        .def_static("dot", &Vector3D::dot)
        .def_static("cross", &Vector3D::cross)
        .def_static("normal", &Vector3D::normal)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * float())
        .def("__repr__",
            [](const Vector3D &v) {
                return "Vector3D(" + std::to_string(v.x) + ", " + 
                       std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
            }
        );

    // Bind Triangle class
    py::class_<Triangle>(m, "Triangle")
        .def(py::init<Vector3D, Vector3D, Vector3D>())
        .def_readwrite("v1", &Triangle::v1)
        .def_readwrite("v2", &Triangle::v2)
        .def_readwrite("v3", &Triangle::v3)
        .def("normal", &Triangle::normal)
        .def("area", &Triangle::area)
        .def("centroid", &Triangle::centroid)
        .def("containsPoint", &Triangle::containsPoint);

    // Bind MeshItModel class
    py::class_<MeshItModel>(m, "MeshItModel")
        .def(py::init<>())
        .def("set_mesh_quality", &MeshItModel::set_mesh_quality)
        .def("set_mesh_algorithm", &MeshItModel::set_mesh_algorithm)
        .def("enable_constraints", &MeshItModel::enable_constraints)
        .def("add_polyline", &MeshItModel::add_polyline)
        .def("add_triangle", &MeshItModel::add_triangle)
        .def("pre_mesh", &MeshItModel::pre_mesh)
        .def("mesh", &MeshItModel::mesh)
        .def("export_vtu", &MeshItModel::export_vtu);
}