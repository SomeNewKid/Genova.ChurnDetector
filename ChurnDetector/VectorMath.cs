

namespace Genova.ChurnDetector;

internal static class VectorMath
{
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vector sizes must match");
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    public static void L2Normalize(Span<float> v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++) sum += (double)v[i] * v[i];
        var norm = Math.Sqrt(sum);
        if (norm <= 0) return;
        var inv = (float)(1.0 / norm);
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }

    public static (int topIdx, int secondIdx, float top, float second) Top2(ReadOnlySpan<float> sims)
    {
        int topIdx = -1, secondIdx = -1;
        float top = float.NegativeInfinity, second = float.NegativeInfinity;
        for (int i = 0; i < sims.Length; i++)
        {
            var s = sims[i];
            if (s > top)
            {
                second = top; secondIdx = topIdx;
                top = s; topIdx = i;
            }
            else if (s > second)
            {
                second = s; secondIdx = i;
            }
        }
        return (topIdx, secondIdx, top, second);
    }
}
