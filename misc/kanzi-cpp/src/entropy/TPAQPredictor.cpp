/*
Copyright 2011-2024 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "TPAQPredictor.hpp"

using namespace kanzi;


TPAQMixer::TPAQMixer()
{
    _pr = 2048;
    _skew = 0;
    _w0 = _w1 = _w2 = _w3 = _w4 = _w5 = _w6 = _w7 = 32768;
    _p0 = _p1 = _p2 = _p3 = _p4 = _p5 = _p6 = _p7 = 0;
    _learnRate = BEGIN_LEARN_RATE;
}

