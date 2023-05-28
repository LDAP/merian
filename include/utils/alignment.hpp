namespace merian {

/* Aligns number to alignment by "rounding up" */
template <typename T, typename A> inline T align_ceil(T number, A alignment) {
    return (number + alignment - 1) & (~(alignment - 1));
}

/* Aligns number to alignment by "rounding down" */
template <typename T, typename A> inline T align_floor(T number, A alignment) {
    return number & ~(alignment - 1);
}

/* Aligns number to alignment by "rounding down" */
template <typename T, typename A> inline T is_aligned(T number, A alignment) {
    return !(number & (alignment - 1));
}

} // namespace merian
