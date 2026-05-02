SetFactory("OpenCASCADE");

ms = 0.3;

L = 10;
W = 1;
H = 1;

Box(1) = {0, 0, 0, L, W, H};

// --- Identify faces by their centroid/normal using geometric queries ---
// For a Box(1) = {0,0,0, 10,1,1} OpenCASCADE creates surfaces with known tags.
// The safest approach: use BoundingBox or point-on-surface to identify each face.

// We identify surfaces by testing which ones lie on the relevant planes.
// OpenCASCADE Box surfaces are created in this order:
//   xmin (x=0), xmax (x=L), ymin (y=0), ymax (y=W), zmin (z=0), zmax (z=H)
// But tags depend on geometry kernel — verify with Gmsh GUI (Tools > Visibility)

// Use point-based identification (robust approach):
s[] = Surface In BoundingBox{-0.01,-0.01,-0.01, 0.01, 1.01, 1.01}; // x=0 face
Physical Surface("DIRICHLET_X0", 2) = { s[] };

s2[] = Surface In BoundingBox{-0.01,-0.01,-0.01, 10.01, 1.01, 0.01}; // z=0 face  
Physical Surface("NEUMANN_Z0", 3) = { s2[] };

Physical Volume("VOLUME", 400) = {1};

Mesh.CharacteristicLengthMin = ms;
Mesh.CharacteristicLengthMax = ms;
Mesh.Algorithm3D = 1;

// Colors for visual inspection
Color Red    { Physical Surface{2}; }  // Dirichlet x=0
Color Green  { Physical Surface{3}; }  // Neumann z=0 (pressure)
Color Yellow { Volume{1}; }