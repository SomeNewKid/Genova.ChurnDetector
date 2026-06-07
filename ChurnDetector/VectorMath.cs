// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

namespace Genova.ChurnDetector;

/// <summary>
/// Provides basic vector operations used by the detector during similarity scoring.
/// </summary>
internal static class VectorMath
{
    /// <summary>
    /// Computes the dot product of two equal-length vectors.
    /// </summary>
    /// <param name="a">The first input vector.</param>
    /// <param name="b">The second input vector.</param>
    /// <returns>The dot product of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="a"/> and <paramref name="b"/> have different lengths.
    /// </exception>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vector sizes must match");
        }

        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>
    /// L2-normalizes a vector in place.
    /// </summary>
    /// <param name="v">The vector to normalize.</param>
    public static void L2Normalize(Span<float> v)
    {
        double sum = 0d;
        for (int i = 0; i < v.Length; i++)
        {
            sum += (double)v[i] * v[i];
        }

        double norm = Math.Sqrt(sum);
        if (norm <= 0d)
        {
            return;
        }

        float inv = (float)(1.0 / norm);
        for (int i = 0; i < v.Length; i++)
        {
            v[i] *= inv;
        }
    }

    /// <summary>
    /// Finds the indices and values of the top two elements in a similarity array.
    /// </summary>
    /// <param name="sims">The array of similarity scores.</param>
    /// <returns>
    /// A tuple containing:
    /// <list type="bullet">
    /// <item><description><c>TopIdx</c>: index of the largest value.</description></item>
    /// <item><description><c>SecondIdx</c>: index of the second-largest value.</description></item>
    /// <item><description><c>Top</c>: the largest value.</description></item>
    /// <item><description><c>Second</c>: the second-largest value.</description></item>
    /// </list>
    /// If <paramref name="sims"/> is empty, indices are <c>-1</c> and values are <see cref="float.NegativeInfinity"/>.
    /// </returns>
    public static (int TopIdx, int SecondIdx, float Top, float Second) Top2(ReadOnlySpan<float> sims)
    {
        int topIdx = -1;
        int secondIdx = -1;
        float top = float.NegativeInfinity;
        float second = float.NegativeInfinity;

        for (int i = 0; i < sims.Length; i++)
        {
            float s = sims[i];
            if (s > top)
            {
                second = top;
                secondIdx = topIdx;
                top = s;
                topIdx = i;
            }
            else if (s > second)
            {
                second = s;
                secondIdx = i;
            }
        }

        return (topIdx, secondIdx, top, second);
    }
}
