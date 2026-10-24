"""Rasterize a filled SVG path.  Implement rasterize(d, width, height, fill_rule) -> list[str].

d is an SVG path string over exactly these commands (UPPER case = absolute, lower case = relative):
   M/m x y                 move to: starts a new subpath.  Extra coordinate pairs after it are implicit L/l.
   L/l x y                 line to (several pairs allowed)
   H/h x                   horizontal line to x
   V/v y                   vertical line to y
   C/c x1 y1 x2 y2 x y     cubic Bezier: control points (x1,y1) and (x2,y2), end point (x,y)
   Q/q x1 y1 x y           quadratic Bezier: control point (x1,y1), end point (x,y)
   A/a rx ry rot large sweep x y   elliptical arc of radii rx, ry, x-axis rotation rot, to end point (x,y)
   Z/z                     close the subpath (back to its start point; the current point becomes that start point)
Every command may repeat with further parameter sets, its letter written only once: "M1 1 L2 2 3 3" is two line-tos
and "C1 2 3 4 5 6 7 8 9 10 11" is two cubic segments (the next parameter set follows immediately, exactly as the
implicit line-tos follow an M).

Numbers: integers or decimals, optional sign, optional exponent ("1e1" = 10, "2E-2" = 0.02), separated by
whitespace and/or commas; a command letter may be followed directly by a number ("M10,10L20 10", "M1e1,1L31,1").
A sign may begin a new number without a separator ("23-0.5" is 23 then -0.5).  The two arc flags are read as
ordinary numbers, so each must be separated from the next number by whitespace or a comma (SVG's compact flag
syntax is not required).  Unknown command letters, a parameter set that is missing numbers, a number with no
command before it, and any character outside this grammar -> ValueError.  An arc's large-arc-flag and sweep-flag
must be exactly 0 or 1; any other value -> ValueError.

Relative commands are relative to the current point (after Z: the subpath's start point).  Every subpath is closed
for filling even without Z.

CURVES AND ARCS ARE TURNED INTO STRAIGHT POINTS BY ONE DOCUMENTED RULE.  Every curved segment - one C or Q
command, or one A command - is flattened into exactly N = 32 equal-parameter straight segments: the segment's
parameter p takes the values p = 1/32, 2/32, ..., 32/32, the curve is evaluated at each p, and the resulting
32 points are appended to the current subpath in that order (the segment's own start point is the current point,
which is already in the subpath).  Points are kept exactly, with no rounding; the flattened points are the only
vertices the filling test below sees.

   Cubic C/c: parameter t in [0,1]; with P0 the current point and P1,P2 the control points, P3 = (x,y),
      B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3.
   Quadratic Q/q: parameter t in [0,1]; with P0 the current point, P1 the control point, P2 = (x,y),
      B(t) = (1-t)^2 P0 + 2(1-t) t P1 + t^2 P2.

   Elliptical arc A/a: rot is the x-axis rotation in DEGREES; the endpoint is (x,y).  Apply the SVG rules:
   (a) if the current point equals the endpoint the arc is empty (no points, the current point is unchanged);
   (b) if rx = 0 or ry = 0, the arc is a straight line to the endpoint (that one point is appended);
   (c) take rx = |rx|, ry = |ry|; rotate both endpoints by -rot about their midpoint: with dx,dy =
       (x1-x2)/2, (y1-y2)/2 and phi = rot in radians,   x1' = cos(phi)*dx + sin(phi)*dy,
       y1' = -sin(phi)*dx + cos(phi)*dy;
   (d) RADIUS CORRECTION: with L = x1'^2/rx^2 + y1'^2/ry^2, if L > 1 then rx <- rx*sqrt(L) and ry <- ry*sqrt(L)
       (the radii are scaled up just enough to reach - never scaled down);
   (e) centre in the rotated frame: c' = s * sqrt( max(0, (rx^2*ry^2 - rx^2*y1'^2 - ry^2*x1'^2) /
       (rx^2*y1'^2 + ry^2*x1'^2)) ) * ( rx*y1'/ry , -ry*x1'/rx ), where s = +1 if large-arc-flag != sweep-flag
       and s = -1 if they are equal; then rotate the centre back about the midpoint:
       cx = cos(phi)*c'x - sin(phi)*c'y + (x1+x2)/2,   cy = sin(phi)*c'x + cos(phi)*c'y + (y1+y2)/2;
   (f) angles: theta1 = angle of the vector (1,0) to ( (x1'-cx')/rx , (y1'-cy')/ry ); the sweep
       delta = angle of ( (x1'-cx')/rx , (y1'-cy')/ry ) to ( (-x1'-cx')/rx , (-y1'-cy')/ry ); then if
       sweep-flag = 0 and delta > 0 subtract 2*pi, and if sweep-flag = 1 and delta < 0 add 2*pi (so sweep 1
       travels in the increasing-angle direction and sweep 0 in the decreasing one).  The angle from u to v is
       sign * acos(clamp((u.v)/(|u|*|v|), -1, 1)) with sign = +1 if the cross product ux*vy - uy*vx >= 0
       else -1;
   (g) the arc parameter is the angle theta = theta1 + p*delta, again at p = 1/32 .. 32/32, and the flattened
       point is ( cx + rx*cos(theta)*cos(phi) - ry*sin(theta)*sin(phi),
                   cy + rx*cos(theta)*sin(phi) + ry*sin(theta)*cos(phi) ).

FILLING.  Coordinates are pixels: x right, y DOWN; pixel (col, row) covers [col, col+1) x [row, row+1).
A pixel is filled iff its CENTRE (col + 0.5, row + 0.5) is inside the shape:
   cast a ray from the centre toward +x; for every edge (x0,y0)->(x1,y1) with (y0 <= cy < y1) or (y1 <= cy < y0)
   (each consecutive pair of a subpath's points, the last joined back to the first) compute the crossing x;
   count it if crossing_x > cx.  Its winding contribution is +1 if y1 > y0 else -1.
   fill_rule "nonzero": inside iff the sum of contributions != 0;  "evenodd": inside iff the crossing count is odd.
Return height strings of width characters: '#' filled, '.' empty.
"""


def rasterize(d, width, height, fill_rule="nonzero"):
    raise NotImplementedError
