SetFactory("OpenCASCADE"); //Necessary for doing the intern subraction

//Given parameters from: "A transmurally heterogeneous orthotropic activation model for ventricular contraction and its numerical validation"

//Ventricle geometry with fiber orientation

//Mesh size
ms = 0.7;

//Focal length 
d = 2.91;
xi_in = 0.6;       // inter ellipsoidale
xi_out = 1.02;     // outer ellipsoidale
z_cut = 1.19;      // base

//Calculate the 4 semi axes of the ellipsoids (2 inner and 2 outer)
R_in_xy = d * Sinh(xi_in);
R_in_z  = d * Cosh(xi_in);

R_out_xy = d * Sinh(xi_out);
R_out_z  = d * Cosh(xi_out);

//Create the outer ellipsoid
v_out = newv; // Create a new volume
Sphere(v_out) = {0, 0, 0, 1}; // Unitary sphere in the origin with tag = newv
Dilate {{0, 0, 0}, {R_out_xy, R_out_xy, R_out_z}} { Volume{v_out}; } 

//Create the inner ellipsoid 
v_in = newv; 
Sphere(v_in) = {0, 0, 0, 1};
Dilate {{0, 0, 0}, {R_in_xy, R_in_xy, R_in_z}} { Volume{v_in}; }

//Create the ventricle by subtracting the inner from the outer ellipsoid
v_wall = newv;
BooleanDifference(v_wall) = { Volume{v_out}; Delete; }{ Volume{v_in}; Delete; };

// Cut the ventricle at the base (z = z_cut)
Box(1000) = {-50, -50, z_cut, 100, 100, 100};

// With the same concept we cancel the upper part of the ventricle
BooleanDifference{ Volume{v_wall}; Delete; }{ Volume{1000}; Delete; }



// For this geometry, after the boolean operations Gmsh creates exactly 3
// boundary surfaces with stable tags:
//   - Surface{2}: base cut (z = z_cut)  -> Dirichlet (id=2)
//   - Surface{3}: endocardium (inner)   -> Neumann  (id=3)
//   - Surface{1}: epicardium (outer)    -> Robin    (id=4)
Physical Surface("BASE", 2)    = { 2 };
Physical Surface("ENDO_IN", 3) = { 3 };
Physical Surface("EPI_OUT", 4) = { 1 };
Physical Volume("VOLUME", 400) = { v_wall };

//Mesh size specification
Mesh.CharacteristicLengthMin = ms;
Mesh.CharacteristicLengthMax = ms;

// Base (Dirichlet, boundary_id=2) -> ROSSO
Color Red { Surface{ 2 }; }

// Endocardio (Neumann, boundary_id=3) -> VERDE
Color Green { Surface{ 3 }; }

// Epicardio (Robin, boundary_id=4) -> VIOLA
Color Purple { Surface{ 1 }; }

// Volume -> Giallo
Color Yellow { Volume{ v_wall }; }