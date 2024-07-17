#ifndef _BIT_TWIDDLING_H_
#define _BIT_TWIDDLING_H_

uint next_power_of_two(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

#endif // _BIT_TWIDDLING_H_
