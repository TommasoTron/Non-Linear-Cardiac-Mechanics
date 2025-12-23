// Unit cube [0,1]^3 with explicit surface ids.

Mesh.MshFileVersion = 2.2;

lc = 0.2 ;
Lx= 10;
Ly=1;
Lz=1;


Point(1) = {0, 0, 0, lc};
Point(2) = {Lx, 0, 0, lc};
Point(3) = {Lx, Ly, 0, lc};
Point(4) = {0, Ly, 0, lc};
Point(5) = {0, 0, Lz, lc};
Point(6) = {Lx, 0, Lz, lc};
Point(7) = {Lx, Ly, Lz, lc};
Point(8) = {0, Ly, Lz, lc};

Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};
Line(5)  = {5, 6};
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Surfaces:
// 1: z=0, 2: z=1, 3: y=0, 4: y=1, 5: x=0, 6: x=1
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

Line Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};

Line Loop(4) = {-3, 11, 7, -12};
Plane Surface(4) = {4};

Line Loop(5) = {-4, 12, 8, -9};
Plane Surface(5) = {5};

Line Loop(6) = {2, 11, -6, -10};
Plane Surface(6) = {6};

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Dirichlet: faces with x=0 or y=0 or z=0  -> surfaces {5,3,1}
// Neumann: the remaining two faces           -> surfaces {2,4}
// Robin: one of the other face              -> surface {6}
//
// Boundary ids:
//   Dirichlet = 2, Neumann = 3, Robin = 4
Physical Surface("Dirichlet", 2) = {1, 3, 5};
Physical Surface("Neumann",   3) = {2, 4};
Physical Surface("Robin",     4) = {6};
Physical Volume("Domain", 1) = {1};
