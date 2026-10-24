SHAPES = [
    # the original line shapes - they must keep passing
    ("star_nonzero", "M16 2 L25 30 L2 12 L30 12 L7 30 Z", 32, 32, "nonzero"),
    ("star_evenodd", "M16 2 L25 30 L2 12 L30 12 L7 30 Z", 32, 32, "evenodd"),
    ("donut_same_dir_nonzero", "M2 2 H30 V30 H2 Z M10 10 H22 V22 H10 Z", 32, 32, "nonzero"),
    ("donut_opposite_nonzero", "M2 2 H30 V30 H2 Z M10 10 V22 H22 V10 Z", 32, 32, "nonzero"),
    ("donut_same_dir_evenodd", "M2 2 H30 V30 H2 Z M10 10 H22 V22 H10 Z", 32, 32, "evenodd"),
    ("relative_arrow", "m4,14 l12,-10 v6 h12 v8 h-12 v6 z", 32, 32, "nonzero"),
    ("implicit_lineto_unclosed", "M3.5 3.5 28.25 6 20 28.75", 32, 32, "nonzero"),
    ("two_subpaths_relative_after_z", "M2 2 h6 v6 h-6 z m10 10 h8 v8 h-8 z", 32, 32, "nonzero"),
    ("self_overlap_twice", "M4 4 H28 V28 H4 Z M4 4 H28 V28 H4 Z", 32, 32, "evenodd"),
    ("compact_numbers", "M1,1L31,1L16,31Z M16,8L23-0.5", 32, 32, "nonzero"),
    # cubic Beziers
    ("cubic_arch_nonzero", "M4 28 C4 6 28 6 28 28 Z", 32, 32, "nonzero"),
    ("cubic_relative_closed", "m6,26 c0,-18 20,-18 20,0 z", 32, 32, "nonzero"),
    ("cubic_repeated_params", "M4 20 C8 4 12 4 16 20 20 4 24 4 28 20 Z", 32, 32, "nonzero"),
    # quadratic Beziers
    ("quad_arch_evenodd", "M4 28 Q16 2 28 28 Z", 32, 32, "evenodd"),
    ("quad_relative_closed", "m6,26 q10,-24 20,0 z", 32, 32, "nonzero"),
    # elliptical arcs
    ("arc_sweep1", "M3 16 A13 13 0 0 1 29 16 Z", 32, 32, "nonzero"),
    ("arc_sweep0", "M3 16 A13 13 0 0 0 29 16 Z", 32, 32, "nonzero"),
    ("arc_large_flag", "M3 16 A20 20 0 1 1 29 16 Z", 32, 32, "nonzero"),
    ("arc_small_flag", "M3 16 A20 20 0 0 1 29 16 Z", 32, 32, "nonzero"),
    ("arc_radius_correction", "M4 16 A2 2 0 0 1 28 16 Z", 32, 32, "nonzero"),
    ("arc_rotated", "M6 7 A14 7 42 1 1 25 19 Z", 32, 32, "nonzero"),
    ("arc_relative", "m3,16 a13 13 0 0 1 26 0 z", 32, 32, "nonzero"),
    # a curved subpath crossing a straight one: nonzero and evenodd must differ
    ("curve_circle_nonzero", "M2 2 H30 V30 H2 Z M16 16 m-9 0 a9 9 0 1 1 18 0 a9 9 0 1 1 -18 0", 32, 32, "nonzero"),
    ("curve_circle_evenodd", "M2 2 H30 V30 H2 Z M16 16 m-9 0 a9 9 0 1 1 18 0 a9 9 0 1 1 -18 0", 32, 32, "evenodd"),
    ("curve_blob_nonzero", "M2 2 H30 V30 H2 Z M16 20 C8 20 8 8 16 8 C24 8 24 20 16 20 Z", 32, 32, "nonzero"),
    ("curve_blob_evenodd", "M2 2 H30 V30 H2 Z M16 20 C8 20 8 8 16 8 C24 8 24 20 16 20 Z", 32, 32, "evenodd"),
    # number parsing: exponent notation + a sign starting a new number
    ("compact_e_notation", "M1e0,1e0 L3.1e1,1e0 L1.6e1,3.1e1 Z M1.4e1,1.2e1 L2.6e1,1.2e1 L2.0e1,2.5e1 Z", 32, 32, "nonzero"),
    # curve and arc in one subpath
    ("combined_curve_arc", "M6 26 Q16 4 26 26 A10 10 0 0 1 6 26 Z", 32, 32, "nonzero"),
]
