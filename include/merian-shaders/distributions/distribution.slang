__exported import "merian-shaders/utils/random.slang";

namespace merian {

struct DistributionSample<ValueType : IFloat> {
    ValueType s;
    float pdf;
}

interface Distribution<ValueType : IFloat> {
    float pdf(const ValueType s);

    ValueType sample(inout RandomGenerator rng);

    DistributionSample<ValueType> sample_eval(inout RandomGenerator rng);
}

}

