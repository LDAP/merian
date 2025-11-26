
// used to define the size of buffer outputs
[__AttributeUsage(_AttributeTargets.Var)]
struct MerianSizeStaticAttribute
{
    int size;
};

// used to define the extent of image and sampler outputs
[__AttributeUsage(_AttributeTargets.Var)]
struct MerianExtentStaticAttribute
{
    int x;
    int y;
    int z;
};

// used to define the size or extent of image or buffer outputs dependent on the size of a fitting input
[__AttributeUsage(_AttributeTargets.Var)]
struct MerianExtentAsAttribute
{
    String name;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct MerianIntRangeAttribute
{
    int min;
    int max;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct MerianFloatRangeAttribute
{
    float min;
    float max;
};