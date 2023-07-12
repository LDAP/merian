Adjust brightness and contrast.

Inputs:

| Type  | Input ID | Input name | Description     | Delay |
|-------|----------|------------|-----------------|-------|
| Image | 0        | in         | input image     | no    |

Outputs:

| Type  | Output ID | Output name | Description         | Format/Resolution        | Persistent |
|-------|-----------|-------------|---------------------|--------------------------|------------|
| Image | 0         | out         | result image        | user defined             | no         |


This is a reimplementation of https://github.com/hanatos/vkdt/tree/master/src/pipe/modules/filmcurv

* `brightness` change the overall brightness of the output. this is the reciprocal lambda parameter of the weibull distribution
* `contrast` contrast of the output. this is the k parameter of the weibull distribution
* `bias` offset to add to black pixels. use to lift for low contrast renders and to correct elevated raw black level
* `colour` how to reconstruct colour after applying the curve to the brightness. darktable ucs has in general better colours, but hsl is more robust for edge cases near black

VKDT is licensed under:

copyright 2019 johannes hanika

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

