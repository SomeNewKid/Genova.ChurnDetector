
using System.Text;

namespace Genova.ChurnDetector;

internal sealed class LocalHashEmbedder
{
    public int Dimension { get; }
    public int Ngram { get; } = 3;
    public bool Signed { get; } = true;
    public int PaddingSpaces { get; } = 2;

    private const ulong FNV_OFFSET = 1469598103934665603UL;
    private const ulong FNV_PRIME = 1099511628211UL;

    public LocalHashEmbedder(int dimension)
    {
        Dimension = dimension;
    }

    public float[] Embed(string raw)
    {
        var text = TextPreprocessor.Clean(raw ?? string.Empty);
        text = new string(' ', PaddingSpaces) + text + new string(' ', PaddingSpaces);

        var vec = new float[Dimension];
        if (text.Length < Ngram) return vec;

        for (int i = 0; i <= text.Length - Ngram; i++)
        {
            var gram = text.AsSpan(i, Ngram);
            var h = Fnv1a64(gram);
            var idx = (int)(h % (ulong)Dimension);
            var val = Signed && ((h & 1UL) == 1UL) ? -1f : 1f;
            vec[idx] += val;
        }

        VectorMath.L2Normalize(vec);
        return vec;
    }

    private static ulong Fnv1a64(ReadOnlySpan<char> span)
    {
        // Encode the whole 3-char gram to UTF-8 in one go, then FNV-1a over the bytes.
        // 16 bytes is plenty (we only use ASCII after preprocessing).
        Span<byte> tmp = stackalloc byte[16];
        int len = Encoding.UTF8.GetBytes(span, tmp);

        const ulong FNV_OFFSET = 1469598103934665603UL;
        const ulong FNV_PRIME = 1099511628211UL;

        ulong h = FNV_OFFSET;
        for (int j = 0; j < len; j++)
        {
            h ^= tmp[j];
            h *= FNV_PRIME;
        }
        return h;
    }
}
