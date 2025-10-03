// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace Genova.ChurnDetector;

/// <summary>
/// Provides a lightweight, deterministic character trigram hashing embedder.
/// </summary>
[SuppressMessage(
    "StyleCop.CSharp.NamingRules",
    "SA1310:Field names should not contain underscore",
    Justification = "Readability of constants.")]
internal sealed class LocalHashEmbedder
{
    private const ulong FNV_OFFSET = 1469598103934665603UL;
    private const ulong FNV_PRIME = 1099511628211UL;

    /// <summary>
    /// Initializes a new instance of the <see cref="LocalHashEmbedder"/> class.
    /// </summary>
    /// <param name="dimension">The embedding dimension.</param>
    public LocalHashEmbedder(int dimension)
    {
        Dimension = dimension;
    }

    /// <summary>
    /// Gets the dimensionality of the output embedding vector.
    /// </summary>
    public int Dimension { get; }

    /// <summary>
    /// Gets the n-gram size used for hashing (fixed at 3 for trigrams).
    /// </summary>
    public int Ngram { get; } = 3;

    /// <summary>
    /// Gets a value indicating whether to apply signed bucket updates (±1) during hashing.
    /// </summary>
    public bool Signed { get; } = true;

    /// <summary>
    /// Gets the number of padding spaces applied at both ends of the input prior to n-gram extraction.
    /// </summary>
    public int PaddingSpaces { get; } = 2;

    /// <summary>
    /// Computes an embedding for the provided text using character trigram hashing.
    /// </summary>
    /// <param name="raw">The input text to embed.</param>
    /// <returns>An L2-normalized embedding vector of length <see cref="Dimension"/>.</returns>
    public float[] Embed(string raw)
    {
        string text = TextPreprocessor.Clean(raw ?? string.Empty);
        text = new string(' ', PaddingSpaces) + text + new string(' ', PaddingSpaces);

        float[] vec = new float[Dimension];
        if (text.Length < Ngram)
        {
            return vec;
        }

        for (int i = 0; i <= text.Length - Ngram; i++)
        {
            ReadOnlySpan<char> gram = text.AsSpan(i, Ngram);
            ulong h = Fnv1a64(gram);
            int idx = (int)(h % (ulong)Dimension);
            float val = (Signed && ((h & 1UL) == 1UL)) ? -1f : 1f;
            vec[idx] += val;
        }

        VectorMath.L2Normalize(vec);
        return vec;
    }

    /// <summary>
    /// Computes the 64-bit FNV-1a hash for a UTF-8–encoded character span.
    /// </summary>
    /// <param name="span">The character span to hash.</param>
    /// <returns>The 64-bit FNV-1a hash.</returns>
    private static ulong Fnv1a64(ReadOnlySpan<char> span)
    {
        // Encode the whole n-gram to UTF-8 in one go, then FNV-1a over the bytes.
        // 16 bytes is sufficient because preprocessing keeps ASCII-range characters.
        Span<byte> tmp = stackalloc byte[16];
        int len = Encoding.UTF8.GetBytes(span, tmp);

        ulong h = FNV_OFFSET;
        for (int j = 0; j < len; j++)
        {
            h ^= tmp[j];
            h *= FNV_PRIME;
        }

        return h;
    }
}
