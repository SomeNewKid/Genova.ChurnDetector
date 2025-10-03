// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Text.Json.Serialization;

namespace Genova.ChurnDetector;

/// <summary>
/// Describes the configuration of the embedding model used by the detector.
/// </summary>
internal sealed class EmbedderInfo
{
    /// <summary>
    /// Gets or sets the embedder type (e.g., <c>CharTrigramHashing</c>).
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = "CharTrigramHashing";

    /// <summary>
    /// Gets or sets the n-gram size used by the embedder (e.g., <c>3</c> for trigrams).
    /// </summary>
    [JsonPropertyName("ngram")]
    public int Ngram { get; set; } = 3;

    /// <summary>
    /// Gets or sets the embedding vector dimension.
    /// </summary>
    [JsonPropertyName("dim")]
    public int Dim { get; set; } = 256;

    /// <summary>
    /// Gets or sets the hashing function identifier (e.g., <c>FNV1A64</c>).
    /// </summary>
    [JsonPropertyName("hash")]
    public string Hash { get; set; } = "FNV1A64";

    /// <summary>
    /// Gets or sets a value indicating whether signed bucket updates are used during hashing.
    /// </summary>
    [JsonPropertyName("signed")]
    public bool Signed { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether input text is lowercased before processing.
    /// </summary>
    [JsonPropertyName("lowercase")]
    public bool Lowercase { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether non-alphanumeric characters (except space) are removed.
    /// </summary>
    [JsonPropertyName("keep_alnum_space_only")]
    public bool KeepAlnumSpaceOnly { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of padding spaces added to the start and end of input prior to n-gram extraction.
    /// </summary>
    [JsonPropertyName("padding_spaces")]
    public int PaddingSpaces { get; set; } = 2;
}
