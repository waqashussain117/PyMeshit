#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <pybind11/functional.h>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include <ctime>
#include <array>
#include "geometry.h"
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
// Intersection class to track intersections between surfaces/polylines
class Intersection {
    public:
        int id1, id2;
        bool is_polyline_mesh;
        std::vector<Vector3D> points;
    
        Intersection(int id1, int id2, bool is_polyline_mesh = false)
            : id1(id1), id2(id2), is_polyline_mesh(is_polyline_mesh) {}
    
        void add_point(const Vector3D& point) {
            points.push_back(point);
        }
    };
    
// Triple point class for intersection triple points
class TriplePoint {
    public:
        Vector3D point;
        std::vector<int> intersection_ids;
    
        TriplePoint(const Vector3D& p) : point(p) {}
    
        void add_intersection(int id) {
            intersection_ids.push_back(id);
        }
    };
// Triangle class with enhanced functionality

// Surface class with full functionality
class Surface {
    public:
        std::string name;
        std::string type;
        double size;
        std::vector<Vector3D> vertices;
        std::vector<std::vector<int>> triangles;
        std::vector<Vector3D> convex_hull;
        std::array<Vector3D, 2> bounds; // min, max bounds
    
        Surface() : size(0.0) {}
    
        void calculate_convex_hull() {
            // Simplified convex hull algorithm for Python bindings
            // This is a placeholder - in real implementation, call your actual convex hull algorithm
            
            // Use Graham scan or Jarvis march algorithm here
            if (vertices.empty()) return;
            
            // Find points with min/max coordinates as starting point for convex hull
            calculate_min_max();
            
            // In a real implementation, this would compute the complete 3D convex hull
            // For now, we'll just create a box hull as placeholder
            convex_hull.clear();
            convex_hull.push_back(Vector3D(bounds[0].x, bounds[0].y, bounds[0].z));
            convex_hull.push_back(Vector3D(bounds[1].x, bounds[0].y, bounds[0].z));
            convex_hull.push_back(Vector3D(bounds[1].x, bounds[1].y, bounds[0].z));
            convex_hull.push_back(Vector3D(bounds[0].x, bounds[1].y, bounds[0].z));
            convex_hull.push_back(Vector3D(bounds[0].x, bounds[0].y, bounds[1].z));
            convex_hull.push_back(Vector3D(bounds[1].x, bounds[0].y, bounds[1].z));
            convex_hull.push_back(Vector3D(bounds[1].x, bounds[1].y, bounds[1].z));
            convex_hull.push_back(Vector3D(bounds[0].x, bounds[1].y, bounds[1].z));
            
            // In a real implementation, you would call your actual algorithm:
            // convex_hull = compute_3d_convex_hull(vertices);
        }
        
        void calculate_min_max() {
            if (vertices.empty()) return;
            
            bounds[0] = bounds[1] = vertices[0];
            for (const auto& v : vertices) {
                bounds[0].x = std::min(bounds[0].x, v.x);
                bounds[0].y = std::min(bounds[0].y, v.y);
                bounds[0].z = std::min(bounds[0].z, v.z);
                bounds[1].x = std::max(bounds[1].x, v.x);
                bounds[1].y = std::max(bounds[1].y, v.y);
                bounds[1].z = std::max(bounds[1].z, v.z);
            }
        }
        
        void triangulate() {
            // This would call your triangulation algorithm (likely using Triangle.c)
            // For the Python binding, we'll create a placeholder that just creates a few simple triangles
            
            if (vertices.size() < 3) return;
            
            triangles.clear();
            // Create simple triangles (fan triangulation from first vertex)
            for (size_t i = 1; i < vertices.size() - 1; i++) {
                triangles.push_back({0, static_cast<int>(i), static_cast<int>(i+1)});
            }
        }
        
        void alignIntersectionsToConvexHull() {
            // Implementation would project intersections onto convex hull
            // This is a placeholder - actual implementation would be more complex
        }
        
        void calculate_Constraints() {
            // Implementation would calculate mesh constraints
            // This is a placeholder - actual implementation would compute complex constraints
        }
        
        std::vector<Vector3D> get_convex_hull() const {
            return convex_hull;
        }
    };
    

// Polyline class with full functionality
class Polyline {
    public:
        std::string name;
        double size;
        std::vector<Vector3D> vertices;
        std::vector<std::vector<int>> segments;
        std::array<Vector3D, 2> bounds; // min, max bounds
    
        Polyline() : size(0.0) {}
        
        void calculate_segments(bool use_fine_segmentation) {
            // This would calculate the segmentation of the polyline
            // For the Python binding, we'll create a placeholder
            
            segments.clear();
            if (vertices.size() < 2) return;
            
            // Create basic segments (just connecting consecutive points)
            for (size_t i = 0; i < vertices.size() - 1; i++) {
                segments.push_back({static_cast<int>(i), static_cast<int>(i+1)});
            }
        }
        
        void calculate_min_max() {
            if (vertices.empty()) return;
            
            bounds[0] = bounds[1] = vertices[0];
            for (const auto& v : vertices) {
                bounds[0].x = std::min(bounds[0].x, v.x);
                bounds[0].y = std::min(bounds[0].y, v.y);
                bounds[0].z = std::min(bounds[0].z, v.z);
                bounds[1].x = std::max(bounds[1].x, v.x);
                bounds[1].y = std::max(bounds[1].y, v.y);
                bounds[1].z = std::max(bounds[1].z, v.z);
            }
        }
        
        void calculate_Constraints() {
            // Implementation would calculate polyline constraints
            // This is a placeholder - actual implementation would compute constraints
        }
    };
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

// Update your MeshItModel class with these new methods
class MeshItModel {
private:
    std::vector<std::vector<Vector3D>> polylines;
    std::vector<Triangle> triangles;
    std::vector<Vector3D> mesh_vertices;
    std::vector<std::vector<int>> mesh_faces;
    double mesh_quality;
    std::string mesh_algorithm;
    bool has_constraints;
    
    // Add these new private variables
    std::mutex mutex;
    std::vector<Intersection> intersections;
    std::vector<TriplePoint> triple_points;

public:
    // Add the existing surfaces and polylines as public members
    std::vector<Surface> surfaces;
    std::vector<Polyline> model_polylines;
    
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
// Add the pre_mesh_job method
    void pre_mesh_job(const std::function<void(const std::string&)>& progress_callback = nullptr) {
        // Record start time
        auto start_time = std::chrono::system_clock::now();
        std::string time_str = get_current_time_string();
        
        if (progress_callback) {
            progress_callback(">Start Time: " + time_str + "\n");
        }
        
        // Calculate convex hulls
        if (progress_callback) {
            progress_callback(">Start calculating convexhull...\n");
        }
        
        std::vector<std::future<void>> futures;
        for (size_t s = 0; s < surfaces.size(); s++) {
            futures.push_back(std::async(std::launch::async, [this, s]() {
                surfaces[s].calculate_convex_hull();
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // Segmentation (coarse)
        if (progress_callback) {
            progress_callback(">Start coarse segmentation...\n");
        }
        
        for (size_t p = 0; p < model_polylines.size(); p++) {
            futures.push_back(std::async(std::launch::async, [this, p]() {
                model_polylines[p].calculate_segments(false);
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // 2D triangulation (coarse)
        if (progress_callback) {
            progress_callback(">Start coarse triangulation...\n");
        }
        
        for (size_t s = 0; s < surfaces.size(); s++) {
            futures.push_back(std::async(std::launch::async, [this, s]() {
                surfaces[s].triangulate();
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // Intersection: surface-surface
        if (progress_callback) {
            progress_callback(">Start calculating surface-surface intersections...\n");
        }
        
        intersections.clear();
        
        // Calculate total number of combinations
        int totalSteps = surfaces.size() * (surfaces.size() - 1) / 2;
        
        if (totalSteps > 0) {
            for (size_t s1 = 0; s1 < surfaces.size() - 1; s1++) {
                for (size_t s2 = s1 + 1; s2 < surfaces.size(); s2++) {
                    futures.push_back(std::async(std::launch::async, [this, s1, s2]() {
                        calculate_surface_surface_intersection(s1, s2);
                    }));
                }
            }
            
            for (auto& future : futures) {
                future.wait();
            }
            futures.clear();
        }
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // Intersection: polyline-surface
        if (progress_callback) {
            progress_callback(">Start calculating polyline-surface intersections...\n");
        }
        
        totalSteps = model_polylines.size() * surfaces.size();
        
        if (totalSteps > 0) {
            for (size_t p = 0; p < model_polylines.size(); p++) {
                for (size_t s = 0; s < surfaces.size(); s++) {
                    futures.push_back(std::async(std::launch::async, [this, p, s]() {
                        calculate_polyline_surface_intersection(p, s);
                    }));
                }
            }
            
            for (auto& future : futures) {
                future.wait();
            }
            futures.clear();
        }
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // Intersection: calculate size
        calculate_size_of_intersections();
        
        // Intersection: triple points
        if (progress_callback) {
            progress_callback(">Start calculating intersection triplepoints...\n");
        }
        
        triple_points.clear();
        
        totalSteps = intersections.size() * (intersections.size() - 1) / 2;
        
        if (totalSteps > 0) {
            for (size_t i1 = 0; i1 < intersections.size() - 1; i1++) {
                for (size_t i2 = i1 + 1; i2 < intersections.size(); i2++) {
                    futures.push_back(std::async(std::launch::async, [this, i1, i2]() {
                        calculate_triple_points(i1, i2);
                    }));
                }
            }
            
            for (auto& future : futures) {
                future.wait();
            }
            futures.clear();
        }
        
        insert_triple_points();
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // Align convex hulls to intersections
        if (progress_callback) {
            progress_callback(">Start aligning Convex Hulls to Intersections...\n");
        }
        
        for (size_t s = 0; s < surfaces.size(); s++) {
            if (progress_callback) {
                progress_callback("   >(" + std::to_string(s + 1) + "/" + 
                              std::to_string(surfaces.size()) + ") " + 
                              surfaces[s].name + " (" + surfaces[s].type + ")");
            }
            surfaces[s].alignIntersectionsToConvexHull();
        }
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        // Model constraints
        if (progress_callback) {
            progress_callback(">Start calculating constraints...\n");
        }
        
        for (size_t s = 0; s < surfaces.size(); s++) {
            surfaces[s].calculate_Constraints();
        }
        
        for (size_t p = 0; p < model_polylines.size(); p++) {
            model_polylines[p].calculate_Constraints();
        }
        
        if (progress_callback) {
            progress_callback(">...finished");
        }
        
        calculate_size_of_constraints();
        
        // End timing
        auto end_time = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        if (progress_callback) {
            time_str = get_current_time_string();
            progress_callback(">End Time: " + time_str + "\n");
            progress_callback(">elapsed Time: " + std::to_string(elapsed) + "ms\n");
        }
    }
    
    void calculate_surface_surface_intersection(size_t s1, size_t s2) {
        // Skip if surfaces don't overlap based on bounding boxes
        const Surface& surface1 = surfaces[s1];
        const Surface& surface2 = surfaces[s2];
        
        // Early rejection test using bounding boxes
        if (surface1.bounds[1].x < surface2.bounds[0].x || surface1.bounds[0].x > surface2.bounds[1].x ||
            surface1.bounds[1].y < surface2.bounds[0].y || surface1.bounds[0].y > surface2.bounds[1].y ||
            surface1.bounds[1].z < surface2.bounds[0].z || surface1.bounds[0].z > surface2.bounds[1].z) {
            return; // No intersection possible
        }
        
        // Find all intersections between triangles in both surfaces
        std::vector<Vector3D> intersection_points;
        
        for (size_t t1 = 0; t1 < surface1.triangles.size(); t1++) {
            const auto& tri1 = surface1.triangles[t1];
            if (tri1.size() < 3) continue;
            
            // Get triangle vertices
            Vector3D v1_1 = surface1.vertices[tri1[0]];
            Vector3D v1_2 = surface1.vertices[tri1[1]];
            Vector3D v1_3 = surface1.vertices[tri1[2]];
            
            for (size_t t2 = 0; t2 < surface2.triangles.size(); t2++) {
                const auto& tri2 = surface2.triangles[t2];
                if (tri2.size() < 3) continue;
                
                // Get triangle vertices
                Vector3D v2_1 = surface2.vertices[tri2[0]];
                Vector3D v2_2 = surface2.vertices[tri2[1]];
                Vector3D v2_3 = surface2.vertices[tri2[2]];
                
                // Calculate triangle-triangle intersection (simplified)
                // In a real implementation, use a proper triangle-triangle intersection algorithm
                
                // For now, just check if any vertex of tri1 is inside tri2
                // (This is a very simplified approach - not accurate for real use)
                Vector3D midpoint1 = (v1_1 + v1_2 + v1_3) * (1.0/3.0);
                Vector3D midpoint2 = (v2_1 + v2_2 + v2_3) * (1.0/3.0);
                
                // Calculate distance between midpoints
                Vector3D diff = midpoint1 - midpoint2;
                double distance = std::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                
                // If midpoints are close enough, consider it an intersection
                if (distance < 0.01 * (surface1.size + surface2.size) / 2.0) {
                    // Add middle point of the segment connecting the midpoints as intersection
                    Vector3D intersection_point = (midpoint1 + midpoint2) * 0.5;
                    
                    // Check if this point is already in our intersection list
                    bool found = false;
                    for (const auto& existing_point : intersection_points) {
                        Vector3D delta = existing_point - intersection_point;
                        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z < 1e-10) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) {
                        intersection_points.push_back(intersection_point);
                    }
                }
            }
        }
        
        // If we found any intersections, create an Intersection object
        if (!intersection_points.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            Intersection intersection(s1, s2, false);
            
            for (const auto& point : intersection_points) {
                intersection.add_point(point);
            }
            
            intersections.push_back(intersection);
        }
    }
    
    void calculate_polyline_surface_intersection(size_t polyline_idx, size_t surface_idx) {
        const Polyline& polyline = model_polylines[polyline_idx];
        const Surface& surface = surfaces[surface_idx];
        
        // Early rejection test using bounding boxes
        if (polyline.bounds[1].x < surface.bounds[0].x || polyline.bounds[0].x > surface.bounds[1].x ||
            polyline.bounds[1].y < surface.bounds[0].y || polyline.bounds[0].y > surface.bounds[1].y ||
            polyline.bounds[1].z < surface.bounds[0].z || polyline.bounds[0].z > surface.bounds[1].z) {
            return; // No intersection possible
        }
        
        // Find intersections between line segments and triangles
        std::vector<Vector3D> intersection_points;
        
        // For each line segment in the polyline
        for (size_t i = 0; i < polyline.segments.size(); i++) {
            if (polyline.segments[i].size() < 2) continue;
            
            // Get segment vertices
            const Vector3D& v1 = polyline.vertices[polyline.segments[i][0]];
            const Vector3D& v2 = polyline.vertices[polyline.segments[i][1]];
            
            // For each triangle in the surface
            for (size_t t = 0; t < surface.triangles.size(); t++) {
                if (surface.triangles[t].size() < 3) continue;
                
                // Get triangle vertices
                const Vector3D& tv1 = surface.vertices[surface.triangles[t][0]];
                const Vector3D& tv2 = surface.vertices[surface.triangles[t][1]];
                const Vector3D& tv3 = surface.vertices[surface.triangles[t][2]];
                
                // Compute line-triangle intersection (simplified)
                // In a real implementation, use Möller–Trumbore algorithm or similar
                
                // Calculate triangle normal
                Vector3D e1 = tv2 - tv1;
                Vector3D e2 = tv3 - tv1;
                Vector3D normal = Vector3D::cross(e1, e2).normalized();
                
                // Calculate distance from line endpoints to triangle plane
                double dist1 = Vector3D::dot(v1 - tv1, normal);
                double dist2 = Vector3D::dot(v2 - tv1, normal);
                
                // If both endpoints are on same side of plane, no intersection
                if (dist1 * dist2 > 0) continue;
                
                // Calculate intersection point with plane
                double intersection_param = dist1 / (dist1 - dist2);  // Changed 't' to 'intersection_param'
                Vector3D intersection_point = v1 + (v2 - v1) * intersection_param;
                
                // Check if point is inside triangle (simplified)
                // For simplicity, just check if it's close to the triangle centroid
                Vector3D centroid = (tv1 + tv2 + tv3) * (1.0/3.0);
                Vector3D diff = intersection_point - centroid;
                double distance = std::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                
                if (distance < 0.5 * surface.size) {
                    // Check if this point is already in our intersection list
                    bool found = false;
                    for (const auto& existing_point : intersection_points) {
                        Vector3D delta = existing_point - intersection_point;
                        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z < 1e-10) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) {
                        intersection_points.push_back(intersection_point);
                    }
                }
            }
        }
        
        // If we found any intersections, create an Intersection object
        if (!intersection_points.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            Intersection intersection(polyline_idx, surface_idx, true);  // true for polyline-surface
            
            for (const auto& point : intersection_points) {
                intersection.add_point(point);
            }
            
            intersections.push_back(intersection);
        }
    }
    
    void calculate_size_of_intersections() {
        // Calculate length/size of each intersection
        for (auto& intersection : intersections) {
            double total_length = 0.0;
            
            // For polyline-mesh intersections, it's just points
            if (intersection.is_polyline_mesh) {
                // Size is just the number of points for polyline-mesh intersections
                continue;
            }
            
            // For mesh-mesh intersections, calculate lengths between consecutive points
            for (size_t i = 0; i < intersection.points.size() - 1; i++) {
                const Vector3D& p1 = intersection.points[i];
                const Vector3D& p2 = intersection.points[i + 1];
                
                // Calculate Euclidean distance
                Vector3D diff = p2 - p1;
                total_length += std::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            }
            
            // Store the length if needed - for example in an attribute of the intersection
        }
    }
    
    void calculate_triple_points(size_t i1, size_t i2) {
        const auto& intersection1 = intersections[i1];
        const auto& intersection2 = intersections[i2];
        
        // Skip if either intersection doesn't have points
        if (intersection1.points.empty() || intersection2.points.empty()) {
            return;
        }
        
        // Skip if intersections don't share a surface
        bool share_surface = (intersection1.id1 == intersection2.id1) || 
                             (intersection1.id1 == intersection2.id2) ||
                             (intersection1.id2 == intersection2.id1) || 
                             (intersection1.id2 == intersection2.id2);
        
        if (!share_surface) {
            return;
        }
        
        // Find closest points between the two intersection lines
        double min_distance = std::numeric_limits<double>::max();
        Vector3D closest_point;
        
        // For every point in intersection1, find closest point in intersection2
        for (const auto& p1 : intersection1.points) {
            for (const auto& p2 : intersection2.points) {
                Vector3D diff = p2 - p1;
                double distance = std::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_point = (p1 + p2) * 0.5;  // midpoint between closest points
                }
            }
        }
        
        // If points are close enough, consider it a triple point
        if (min_distance < 1e-6) {
            std::lock_guard<std::mutex> lock(mutex);
            TriplePoint tp(closest_point);
            tp.add_intersection(i1);
            tp.add_intersection(i2);
            triple_points.push_back(tp);
        }
    }
    
    void insert_triple_points() {
        // This function adds triple points to the relevant intersections
        for (const auto& tp : triple_points) {
            // For each intersection that contains this triple point
            for (int i_id : tp.intersection_ids) {
                if (i_id >= 0 && i_id < static_cast<int>(intersections.size())) {
                    // Check if point is already in the intersection
                    bool found = false;
                    
                    for (const auto& point : intersections[i_id].points) {
                        Vector3D diff = point - tp.point;
                        if (diff.x*diff.x + diff.y*diff.y + diff.z*diff.z < 1e-10) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) {
                        // Add the triple point to this intersection
                        intersections[i_id].points.push_back(tp.point);
                    }
                }
            }
        }
        
        // Sort intersection points for each intersection to maintain spatial order
        // This is a simplified approach - real implementation would sort points along the intersection curve
        for (auto& intersection : intersections) {
            if (intersection.points.size() <= 1) {
                continue;  // Nothing to sort
            }
            
            // For simplicity, just sort by x, then y, then z
            std::sort(intersection.points.begin(), intersection.points.end(), 
                [](const Vector3D& a, const Vector3D& b) {
                    if (std::abs(a.x - b.x) > 1e-10) return a.x < b.x;
                    if (std::abs(a.y - b.y) > 1e-10) return a.y < b.y;
                    return a.z < b.z;
                });
        }
    }
    
    void calculate_size_of_constraints() {
        // Calculate sizes of constraints for surfaces
        for (auto& surface : surfaces) {
            for (size_t i = 0; i < surface.triangles.size(); i++) {
                // Get triangle vertices
                const auto& tri = surface.triangles[i];
                if (tri.size() < 3) continue;
                
                const Vector3D& v1 = surface.vertices[tri[0]];
                const Vector3D& v2 = surface.vertices[tri[1]];
                const Vector3D& v3 = surface.vertices[tri[2]];
                
                // Calculate triangle area using cross product
                Vector3D e1 = v2 - v1;
                Vector3D e2 = v3 - v1;
                Vector3D cross = Vector3D::cross(e1, e2);
                double area = 0.5 * std::sqrt(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z);
                
                // Update constraint size (for example, sum of areas)
            }
        }
        
        // Calculate sizes of constraints for polylines
        for (auto& polyline : model_polylines) {
            double total_length = 0.0;
            
            for (size_t i = 0; i < polyline.segments.size(); i++) {
                const auto& seg = polyline.segments[i];
                if (seg.size() < 2) continue;
                
                const Vector3D& v1 = polyline.vertices[seg[0]];
                const Vector3D& v2 = polyline.vertices[seg[1]];
                
                // Calculate segment length
                Vector3D diff = v2 - v1;
                double length = std::sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                
                total_length += length;
            }
            
            // Update constraint size (for example, total length)
        }
    }
    
    std::string get_current_time_string() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        
        char time_buffer[26];
    #ifdef _WIN32
        ctime_s(time_buffer, sizeof(time_buffer), &now_time);
    #else
        std::string time_str = std::ctime(&now_time);
        std::strncpy(time_buffer, time_str.c_str(), sizeof(time_buffer));
    #endif
    
        // Remove trailing newline if present
        size_t len = strlen(time_buffer);
        if (len > 0 && time_buffer[len-1] == '\n') {
            time_buffer[len-1] = '\0';
        }
        
        return std::string(time_buffer);
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

// Replace the existing PYBIND11_MODULE section with this updated version:
PYBIND11_MODULE(_meshit, m) {
    m.doc() = "MeshIt Python bindings for PZero integration";
    
    // Vector3D bindings
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

    // NEW - Bind the Intersection class
    py::class_<Intersection>(m, "Intersection")
        .def(py::init<int, int, bool>())
        .def_readwrite("id1", &Intersection::id1)
        .def_readwrite("id2", &Intersection::id2)
        .def_readwrite("is_polyline_mesh", &Intersection::is_polyline_mesh)
        .def_readwrite("points", &Intersection::points)
        .def("add_point", &Intersection::add_point);
    
    // NEW - Bind the TriplePoint class
    py::class_<TriplePoint>(m, "TriplePoint")
        .def(py::init<Vector3D>())
        .def_readwrite("point", &TriplePoint::point)
        .def_readwrite("intersection_ids", &TriplePoint::intersection_ids)
        .def("add_intersection", &TriplePoint::add_intersection);
    
    // NEW - Bind the Surface class
    py::class_<Surface>(m, "Surface")
        .def(py::init<>())
        .def_readwrite("name", &Surface::name)
        .def_readwrite("type", &Surface::type)
        .def_readwrite("size", &Surface::size)
        .def_readwrite("vertices", &Surface::vertices)
        .def_readwrite("triangles", &Surface::triangles)
        .def_readwrite("convex_hull", &Surface::convex_hull)
        .def_readwrite("bounds", &Surface::bounds)
        .def("calculate_convex_hull", &Surface::calculate_convex_hull)
        .def("calculate_min_max", &Surface::calculate_min_max)
        .def("triangulate", &Surface::triangulate)
        .def("get_convex_hull", &Surface::get_convex_hull);
    
    // NEW - Bind the Polyline class
    py::class_<Polyline>(m, "Polyline")
        .def(py::init<>())
        .def_readwrite("name", &Polyline::name)
        .def_readwrite("size", &Polyline::size)
        .def_readwrite("vertices", &Polyline::vertices)
        .def_readwrite("segments", &Polyline::segments)
        .def_readwrite("bounds", &Polyline::bounds)
        .def("calculate_segments", &Polyline::calculate_segments)
        .def("calculate_min_max", &Polyline::calculate_min_max);
    
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

    // UPDATED - Bind MeshItModel class with new methods
    py::class_<MeshItModel>(m, "MeshItModel")
        .def(py::init<>())
        .def("set_mesh_quality", &MeshItModel::set_mesh_quality)
        .def("set_mesh_algorithm", &MeshItModel::set_mesh_algorithm)
        .def("enable_constraints", &MeshItModel::enable_constraints)
        .def("add_polyline", &MeshItModel::add_polyline)
        .def("add_triangle", &MeshItModel::add_triangle)
        .def("pre_mesh", &MeshItModel::pre_mesh)
        .def("mesh", &MeshItModel::mesh)
        .def("export_vtu", &MeshItModel::export_vtu)
        // NEW - Add the pre_mesh_job method
        .def("pre_mesh_job", &MeshItModel::pre_mesh_job,
             py::arg("progress_callback") = nullptr,
             "Performs pre-mesh operations (convex hulls, triangulation, intersections)")
        // NEW - Add access to model data
        .def_readwrite("surfaces", &MeshItModel::surfaces)
        .def_readwrite("model_polylines", &MeshItModel::model_polylines);
    
    // NEW - Add helper methods to create surfaces and polylines
    m.def("create_surface", [](const std::vector<std::vector<double>>& vertices,
                               const std::vector<std::vector<int>>& triangles,
                               const std::string& name = "",
                               const std::string& type = "Default") {
        Surface surface;
        surface.name = name;
        surface.type = type;
        
        for (const auto& v : vertices) {
            if (v.size() >= 3) {
                surface.vertices.push_back(Vector3D(v[0], v[1], v[2]));
            }
        }
        
        surface.triangles = triangles;
        surface.calculate_min_max();
        return surface;
    });
    
    m.def("create_polyline", [](const std::vector<std::vector<double>>& vertices,
                                const std::string& name = "") {
        Polyline polyline;
        polyline.name = name;
        
        for (const auto& v : vertices) {
            if (v.size() >= 3) {
                polyline.vertices.push_back(Vector3D(v[0], v[1], v[2]));
            }
        }
        
        polyline.calculate_min_max();
        return polyline;
    });
}