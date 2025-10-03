// This file is part of the Genova project licensed under the GNU General Public License v3.0.
// See the LICENSE file in the project root for more information.

using System.Text.Json.Serialization;

namespace Genova.ChurnDetector;

/// <summary>
/// Represents the model manifest metadata used by the detector at runtime.
/// </summary>
internal sealed class Manifest
{
    /// <summary>
    /// Gets or sets the manifest schema version.
    /// </summary>
    [JsonPropertyName("version")]
    public int Version { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp (ISO 8601) when the manifest was created.
    /// </summary>
    [JsonPropertyName("created_tuc")]
    public string? CreatedUtc { get; set; }

    /// <summary>
    /// Gets or sets the embedding vector dimensionality expected by the model.
    /// </summary>
    [JsonPropertyName("vector_dim")]
    public int VectorDim { get; set; }

    /// <summary>
    /// Gets or sets the set of labels present in the centroids file (e.g., "churn", "keep", "unsure").
    /// </summary>
    [JsonPropertyName("labels")]
    public string[] Labels { get; set; } = [];

    /// <summary>
    /// Gets or sets the scoring thresholds used by the detector.
    /// </summary>
    [JsonPropertyName("thresholds")]
    public Thresholds Thresholds { get; set; } = new ();

    /// <summary>
    /// Gets or sets the embedder configuration metadata.
    /// </summary>
    [JsonPropertyName("embedder")]
    public EmbedderInfo Embedder { get; set; } = new ();
}
