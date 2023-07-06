#ifndef _COLORS_DNG_H_
#define _COLORS_DNG_H_

// apparently this is what the dng spec says should be used to correct colour
// after application of tone-changing operators (curve). with all the sorting,
// it's based on hsv and designed to keep hue constant in that space.
vec3 // returns hue mapped rgb colour
adjust_colour_dng(
    vec3 col0, // original colour
    vec3 col1) // mapped colour, for instance by rgb curves
{
  bvec3 flip = bvec3(false); // bubble sort
  if(col0.b > col0.g) { col0.bg = col0.gb; col1.bg = col1.gb; flip.x = true; }
  if(col0.g > col0.r) { col0.rg = col0.gr; col1.rg = col1.gr; flip.y = true; }
  if(col0.b > col0.g) { col0.bg = col0.gb; col1.bg = col1.gb; flip.z = true; }

  col1.g = mix(col1.b, col1.r, (col0.g-col0.b+1e-8)/(col0.r-col0.b+1e-8));

  if(flip.z) col1.bg = col1.gb;
  if(flip.y) col1.rg = col1.gr;
  if(flip.x) col1.bg = col1.gb;
  return col1;
}

#endif
