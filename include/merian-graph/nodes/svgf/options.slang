namespace merian {

enum SVGFFilter {
    Atrous,
    Box,
    Subsampled,
};

extern static const SVGFFilter svgf_filter = SVGFFilter.Subsampled;
extern static const bool svgf_kaleidoscope = false;
extern static const bool svgf_kaleidoscope_use_shmem = true;

}
