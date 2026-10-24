"""A tiny deterministic ray tracer.  Implement render() EXACTLY as specified - images are compared pixel by pixel.

Scene (a dict, see SCENE in test_raytrace.py):
  "camera": {"pos": [x, y, z], "fov_deg": f}   looks along -z, +y is up, +x is right; no rotation.
  "spheres": [{"center": [x,y,z], "radius": r, "color": [r,g,b] in 0..1, "reflect": k in 0..1}, ...]
  "plane": {"y": h, "colors": [[r,g,b], [r,g,b]], "reflect": k}   optional infinite plane y = h, a checkerboard
  "light": {"pos": [x,y,z], "intensity": i}   one point light, white
  "ambient": a

Rays: pixel (px, py), px = 0..width-1 left to right, py = 0..height-1 TOP to bottom.  Sample the pixel CENTER:
  ndc_x = (px + 0.5) / width * 2 - 1,  ndc_y = 1 - (py + 0.5) / height * 2
  scale = tan(fov_deg / 2 in radians);  aspect = width / height
  direction = normalize([ndc_x * scale * aspect, ndc_y * scale, -1])   from camera pos.
Hits: the nearest hit with t > 1e-4.  Sphere normal = (hit - center) / radius; plane normal = (0, 1, 0).
Plane checker: cell = floor(hit.x) + floor(hit.z)  (math.floor - mind negative coordinates); colors[cell % 2]
  (Python's %, so the result is 0 or 1 for negative cells too).
Shading (colour c at a hit point p with normal n):
  L = normalize(light.pos - p);  diffuse = max(0, n . L) * light.intensity
  shadow: cast a ray from p + n * 1e-4 toward the light; if it hits any object with 1e-4 < t < distance(p, light),
          diffuse = 0
  local = c * (ambient + diffuse)        (per channel)
  reflection (k = the object's "reflect", depth < 3 bounces): r = d - 2 (d . n) n;  trace from p + n * 1e-4 along r
          result = local * (1 - k) + traced * k          (k = 0: no reflection ray)
Miss (sky): t = 0.5 * (d.y + 1);  sky = (1 - t) * [1, 1, 1] + t * [0.5, 0.7, 1.0]   (no shading)
A reflection ray at depth 3 returns black [0, 0, 0] instead of tracing.
Output: list of height rows, each a list of width (r, g, b) tuples of ints: round(min(1, max(0, channel)) * 255).
"""


def render(scene, width, height):
    raise NotImplementedError
