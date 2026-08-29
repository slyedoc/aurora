#ifndef H_UI_COLOR
#define H_UI_COLOR

// Port of bevy_render's `color_operations.wesl`: the color-space conversions and hue-aware
// mixes the UI gradient shader needs.

const float UI_PI = 3.141592653589793;
const float UI_PI_2 = 6.283185307179586;
const float HUE_GUARD = 0.0001;

// https://en.wikipedia.org/wiki/SRGB
float srgb_gamma(float value) {
  if (value <= 0.0) {
    return value;
  }
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

float srgb_inverse_gamma(float value) {
  if (value <= 0.0) {
    return value;
  }
  if (value <= 0.0031308) {
    return value * 12.92;
  }
  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

vec3 srgb_to_linear_rgb(vec3 c) {
  return vec3(srgb_gamma(c.x), srgb_gamma(c.y), srgb_gamma(c.z));
}

vec3 linear_rgb_to_srgb(vec3 c) {
  return vec3(srgb_inverse_gamma(c.x), srgb_inverse_gamma(c.y), srgb_inverse_gamma(c.z));
}

// https://bottosson.github.io/posts/oklab/
vec3 oklab_to_linear_rgb(vec3 c) {
  float l_ = c.x + 0.39633778 * c.y + 0.21580376 * c.z;
  float m_ = c.x - 0.105561346 * c.y - 0.06385417 * c.z;
  float s_ = c.x - 0.08948418 * c.y - 1.2914855 * c.z;
  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;
  return vec3(
    4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
    -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
    -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s
  );
}

vec3 oklch_to_linear_rgb(vec3 c) {
  float hue = c.z * UI_PI_2;
  return oklab_to_linear_rgb(vec3(c.x, c.y * cos(hue), c.y * sin(hue)));
}

vec3 hsl_to_linear_rgb(vec3 hsl) {
  float h = hsl.x;
  float s = hsl.y;
  float l = hsl.z;
  float c = (1.0 - abs(2.0 * l - 1.0)) * s;
  float hp = h * 6.0;
  float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
  float r = 0.0, g = 0.0, b = 0.0;
  if (0.0 <= hp && hp < 1.0) { r = c; g = x; }
  else if (hp < 2.0) { r = x; g = c; }
  else if (hp < 3.0) { g = c; b = x; }
  else if (hp < 4.0) { g = x; b = c; }
  else if (hp < 5.0) { r = x; b = c; }
  else if (hp < 6.0) { r = c; b = x; }
  float m = l - 0.5 * c;
  return srgb_to_linear_rgb(vec3(r + m, g + m, b + m));
}

vec3 hsv_to_linear_rgb(vec3 hsv) {
  float h = hsv.x * 6.0;
  float s = hsv.y;
  float v = hsv.z;
  float c = v * s;
  float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
  float m = v - c;
  float r = 0.0, g = 0.0, b = 0.0;
  if (0.0 <= h && h < 1.0) { r = c; g = x; }
  else if (h < 2.0) { r = x; g = c; }
  else if (h < 3.0) { g = c; b = x; }
  else if (h < 4.0) { g = x; b = c; }
  else if (h < 5.0) { r = x; b = c; }
  else if (h < 6.0) { r = c; b = x; }
  return srgb_to_linear_rgb(vec3(r + m, g + m, b + m));
}

// --- OKHSL (ported from bevy_color/src/okcolor_convert.rs via color_operations.wesl) ---

float okhsl_toe_inv(float x) {
  const float k_1 = 0.206;
  const float k_2 = 0.03;
  const float k_3 = (1.0 + k_1) / (1.0 + k_2);
  return (x * x + k_1 * x) / (k_3 * (x + k_2));
}

vec2 okhsl_to_ST(vec2 cusp) {
  return vec2(cusp.y / cusp.x, cusp.y / (1.0 - cusp.x));
}

vec2 okhsl_get_ST_mid(float a_, float b_) {
  float S = 0.11516993
    + 1.0 / (7.4477897
      + 4.1590123 * b_
      + a_ * (-2.1955736
        + 1.751984 * b_
        + a_ * (-2.1370494 - 10.02301 * b_
          + a_ * (-4.2489457 + 5.387708 * b_ + 4.69891 * a_))));
  float T = 0.11239642
    + 1.0 / (1.6132032 - 0.6812438 * b_
      + a_ * (0.40370612
        + 0.9014812 * b_
        + a_ * (-0.27087943
          + 0.6122399 * b_
          + a_ * (0.00299215 - 0.45399568 * b_ - 0.14661872 * a_))));
  return vec2(S, T);
}

float okhsl_compute_max_saturation(float a, float b) {
  float k0, k1, k2, k3, k4, wl, wm, ws;
  if (-1.8817033 * a - 0.8093649 * b > 1.0) {
    k0 = 1.1908628; k1 = 1.7657673; k2 = 0.5966264; k3 = 0.755152; k4 = 0.5677124;
    wl = 4.0767417; wm = -3.3077116; ws = 0.23096994;
  } else if (1.8144411 * a - 1.1944528 * b > 1.0) {
    k0 = 0.73956515; k1 = -0.45954404; k2 = 0.08285427; k3 = 0.1254107; k4 = 0.14503204;
    wl = -1.268438; wm = 2.6097574; ws = -0.34131938;
  } else {
    k0 = 1.3573365; k1 = -0.00915799; k2 = -1.1513021; k3 = -0.50559606; k4 = 0.00692167;
    wl = -0.0041960863; wm = -0.7034186; ws = 1.7076147;
  }
  float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

  float k_l = 0.39633778 * a + 0.21580376 * b;
  float k_m = -0.105561346 * a - 0.06385417 * b;
  float k_s = -0.08948418 * a - 1.2914855 * b;

  float l_ = 1.0 + S * k_l;
  float m_ = 1.0 + S * k_m;
  float s_ = 1.0 + S * k_s;

  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;

  float l_dS = 3.0 * k_l * l_ * l_;
  float m_dS = 3.0 * k_m * m_ * m_;
  float s_dS = 3.0 * k_s * s_ * s_;

  float l_dS2 = 6.0 * k_l * k_l * l_;
  float m_dS2 = 6.0 * k_m * k_m * m_;
  float s_dS2 = 6.0 * k_s * k_s * s_;

  float f = wl * l + wm * m + ws * s;
  float f1 = wl * l_dS + wm * m_dS + ws * s_dS;
  float f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2;

  return S - f * f1 / (f1 * f1 - 0.5 * f * f2);
}

vec2 okhsl_find_cusp(float a, float b) {
  float S_cusp = okhsl_compute_max_saturation(a, b);
  vec3 rgb_at_max = oklab_to_linear_rgb(vec3(1.0, S_cusp * a, S_cusp * b));
  float L_cusp = pow(1.0 / max(rgb_at_max.r, max(rgb_at_max.g, rgb_at_max.b)), 1.0 / 3.0);
  return vec2(L_cusp, L_cusp * S_cusp);
}

float okhsl_find_gamut_intersection(float a, float b, float L1, float C1, float L0, vec2 cusp) {
  float cusp_L = cusp.x;
  float cusp_C = cusp.y;
  float t;
  if (((L1 - L0) * cusp_C - (cusp_L - L0) * C1) <= 0.0) {
    t = cusp_C * L0 / (C1 * cusp_L + cusp_C * (L0 - L1));
  } else {
    t = cusp_C * (L0 - 1.0) / (C1 * (cusp_L - 1.0) + cusp_C * (L0 - L1));

    float dL = L1 - L0;
    float dC = C1;

    float k_l = 0.39633778 * a + 0.21580376 * b;
    float k_m = -0.105561346 * a - 0.06385417 * b;
    float k_s = -0.08948418 * a - 1.2914855 * b;

    float l_dt = dL + dC * k_l;
    float m_dt = dL + dC * k_m;
    float s_dt = dL + dC * k_s;

    {
      float L = L0 * (1.0 - t) + t * L1;
      float C = t * C1;

      float l_ = L + C * k_l;
      float m_ = L + C * k_m;
      float s_ = L + C * k_s;

      float l = l_ * l_ * l_;
      float m = m_ * m_ * m_;
      float s = s_ * s_ * s_;

      float ldt = 3.0 * l_dt * l_ * l_;
      float mdt = 3.0 * m_dt * m_ * m_;
      float sdt = 3.0 * s_dt * s_ * s_;

      float ldt2 = 6.0 * l_dt * l_dt * l_;
      float mdt2 = 6.0 * m_dt * m_dt * m_;
      float sdt2 = 6.0 * s_dt * s_dt * s_;

      float r = 4.0767417 * l - 3.3077116 * m + 0.23096994 * s - 1.0;
      float r1 = 4.0767417 * ldt - 3.3077116 * mdt + 0.23096994 * sdt;
      float r2 = 4.0767417 * ldt2 - 3.3077116 * mdt2 + 0.23096994 * sdt2;
      float u_r = r1 / (r1 * r1 - 0.5 * r * r2);
      float t_r = u_r >= 0.0 ? -r * u_r : 3.40282347e+38;

      float g = -1.268438 * l + 2.6097574 * m - 0.34131938 * s - 1.0;
      float g1 = -1.268438 * ldt + 2.6097574 * mdt - 0.34131938 * sdt;
      float g2 = -1.268438 * ldt2 + 2.6097574 * mdt2 - 0.34131938 * sdt2;
      float u_g = g1 / (g1 * g1 - 0.5 * g * g2);
      float t_g = u_g >= 0.0 ? -g * u_g : 3.40282347e+38;

      float b_val = -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s - 1.0;
      float b1 = -0.0041960863 * ldt - 0.7034186 * mdt + 1.7076147 * sdt;
      float b2 = -0.0041960863 * ldt2 - 0.7034186 * mdt2 + 1.7076147 * sdt2;
      float u_b = b1 / (b1 * b1 - 0.5 * b_val * b2);
      float t_b = u_b >= 0.0 ? -b_val * u_b : 3.40282347e+38;

      t = t + min(t_r, min(t_g, t_b));
    }
  }
  return t;
}

vec3 okhsl_get_Cs(float L, float a_, float b_) {
  vec2 cusp = okhsl_find_cusp(a_, b_);
  float C_max = okhsl_find_gamut_intersection(a_, b_, L, 1.0, L, cusp);
  vec2 ST_max = okhsl_to_ST(cusp);

  float k = C_max / min(L * ST_max.x, (1.0 - L) * ST_max.y);

  vec2 ST_mid = okhsl_get_ST_mid(a_, b_);
  float C_a = L * ST_mid.x;
  float C_b = (1.0 - L) * ST_mid.y;
  float C_mid = 0.9 * k * sqrt(sqrt(1.0 / (1.0 / (C_a * C_a * C_a * C_a) + 1.0 / (C_b * C_b * C_b * C_b))));

  float C_0_a = L * 0.4;
  float C_0_b = (1.0 - L) * 0.8;
  float C_0 = sqrt(1.0 / (1.0 / (C_0_a * C_0_a) + 1.0 / (C_0_b * C_0_b)));

  return vec3(C_0, C_mid, C_max);
}

vec3 okhsl_to_oklab(vec3 okhsl) {
  float h = okhsl.x;
  float s = okhsl.y;
  float l = okhsl.z;
  if (l >= 1.0) {
    return vec3(1.0, 0.0, 0.0);
  }
  if (l <= 0.0) {
    return vec3(0.0);
  }
  float a_ = cos(2.0 * UI_PI * h);
  float b_ = sin(2.0 * UI_PI * h);
  float L = okhsl_toe_inv(l);

  vec3 cs = okhsl_get_Cs(L, a_, b_);
  float C_0 = cs.x;
  float C_mid = cs.y;
  float C_max = cs.z;

  const float mid = 0.8;
  const float mid_inv = 1.25;
  float C, t, k_0, k_1, k_2;
  if (s < mid) {
    t = mid_inv * s;
    k_1 = mid * C_0;
    k_2 = 1.0 - k_1 / C_mid;
    C = t * k_1 / (1.0 - k_2 * t);
  } else {
    t = (s - mid) / (1.0 - mid);
    k_0 = C_mid;
    k_1 = (1.0 - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
    k_2 = 1.0 - k_1 / (C_max - C_mid);
    C = k_0 + t * k_1 / (1.0 - k_2 * t);
  }
  return vec3(L, C * a_, C * b_);
}

vec3 okhsl_to_linear_rgb(vec3 okhsl) {
  return oklab_to_linear_rgb(okhsl_to_oklab(okhsl));
}

// --- hue-aware mixes; hues are normalized to [0, 1) ---

// oklch: hue in .z, chroma in .y
vec3 mix_oklch(vec3 a, vec3 b, float t) {
  float h = a.z;
  float g = b.z;
  if (a.y < HUE_GUARD) { h = g; } else if (b.y < HUE_GUARD) { g = h; }
  float hue_diff = g - h;
  if (abs(hue_diff) > 0.5) {
    h += (hue_diff > 0.0 ? hue_diff - 1.0 : hue_diff + 1.0) * t;
  } else {
    h += hue_diff * t;
  }
  return vec3(mix(a.x, b.x, t), mix(a.y, b.y, t), fract(h));
}

vec3 mix_oklch_long(vec3 a, vec3 b, float t) {
  float h = a.z;
  float g = b.z;
  if (a.y < HUE_GUARD) { h = g; } else if (b.y < HUE_GUARD) { g = h; }
  float hue_diff = g - h;
  if (abs(hue_diff) < 0.5) {
    h += (hue_diff >= 0.0 ? hue_diff - 1.0 : hue_diff + 1.0) * t;
  } else {
    h += hue_diff * t;
  }
  return vec3(mix(a.x, b.x, t), mix(a.y, b.y, t), fract(h));
}

// hsl / hsv / okhsl: hue in .x, saturation in .y
vec3 mix_hue_short(vec3 a, vec3 b, float t) {
  float h = a.x;
  float g = b.x;
  if (a.y < HUE_GUARD) { h = g; } else if (b.y < HUE_GUARD) { g = h; }
  return vec3(fract(h + (fract(g - h + 0.5) - 0.5) * t), mix(a.y, b.y, t), mix(a.z, b.z, t));
}

vec3 mix_hue_long(vec3 a, vec3 b, float t) {
  float h = a.x;
  float g = b.x;
  if (a.y < HUE_GUARD) { h = g; } else if (b.y < HUE_GUARD) { g = h; }
  float d = fract(g - h + 0.5) - 0.5;
  return vec3(fract(h + (d + (0.0 < d ? -1.0 : 1.0)) * t), mix(a.y, b.y, t), mix(a.z, b.z, t));
}

vec3 mix_hsv(vec3 a, vec3 b, float t) {
  float h = a.x;
  float g = b.x;
  if (a.y < HUE_GUARD) { h = g; } else if (b.y < HUE_GUARD) { g = h; }
  float hue_diff = g - h;
  if (abs(hue_diff) > 0.5) {
    h += (hue_diff > 0.0 ? hue_diff - 1.0 : hue_diff + 1.0) * t;
  } else {
    h += hue_diff * t;
  }
  return vec3(fract(h), mix(a.y, b.y, t), mix(a.z, b.z, t));
}

vec3 mix_hsv_long(vec3 a, vec3 b, float t) {
  float h = a.x;
  float g = b.x;
  if (a.y < HUE_GUARD) { h = g; } else if (b.y < HUE_GUARD) { g = h; }
  float hue_diff = g - h;
  if (abs(hue_diff) < 0.5) {
    h += (hue_diff >= 0.0 ? hue_diff - 1.0 : hue_diff + 1.0) * t;
  } else {
    h += hue_diff * t;
  }
  return vec3(fract(h), mix(a.y, b.y, t), mix(a.z, b.z, t));
}

#endif
